from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt


def logger(path: Path):
    log=logging.getLogger('stp'); log.setLevel(logging.INFO); log.handlers.clear()
    fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    sh=logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
    path.parent.mkdir(parents=True,exist_ok=True); fh=logging.FileHandler(path,encoding='utf-8'); fh.setFormatter(fmt); log.addHandler(fh)
    return log


def validate_alignment(paths, template, out_csv):
    with rasterio.open(template) as t:
        ref=(str(t.crs),t.width,t.height,tuple(t.transform))
    rows=[]
    for p in paths:
        rec={'path':str(p),'exists':Path(p).exists(),'aligned':False}
        if Path(p).exists():
            with rasterio.open(p) as s:
                cur=(str(s.crs),s.width,s.height,tuple(s.transform)); rec.update({'crs':str(s.crs),'width':s.width,'height':s.height,'nodata':s.nodata,'aligned':cur==ref})
        rows.append(rec)
    df=pd.DataFrame(rows); out_csv.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out_csv,index=False)
    if not df['aligned'].all(): raise ValueError('Raster alignment failed; inspect '+str(out_csv))
    return df


def prepare_vector(path, layer, id_field, name_field, seq_field, extra_fields=None):
    gdf=gpd.read_file(path,layer=layer) if layer else gpd.read_file(path)
    for f in [id_field,name_field]:
        if f not in gdf.columns: raise KeyError(f'Missing field {f} in {path}')
    if gdf[id_field].isna().any(): raise ValueError(f'Null {id_field}')
    gdf=gdf[gdf.geometry.notna()].copy(); gdf['geometry']=gdf.geometry.make_valid()
    if gdf[id_field].duplicated().any():
        keep=[c for c in (extra_fields or []) if c in gdf.columns]
        agg={name_field:'first',**{c:'first' for c in keep}}
        gdf=gdf.dissolve(by=id_field,as_index=False,aggfunc=agg)
    gdf[seq_field]=np.arange(1,len(gdf)+1,dtype=np.int32)
    fields=[seq_field,id_field,name_field]+[c for c in (extra_fields or []) if c in gdf.columns]
    return gdf,gdf[fields].copy()


def rasterize_gdf(gdf, value_field, template, output):
    with rasterio.open(template) as t:
        if gdf.crs!=t.crs: gdf=gdf.to_crs(t.crs)
        arr=rasterize(((g,int(v)) for g,v in zip(gdf.geometry,gdf[value_field]) if g and not g.is_empty),out_shape=(t.height,t.width),transform=t.transform,fill=0,dtype='int32')
        prof=t.profile.copy(); prof.update(dtype='int32',count=1,nodata=0,compress='deflate')
        output.parent.mkdir(parents=True,exist_ok=True)
        with rasterio.open(output,'w',**prof) as dst: dst.write(arr,1)


def create_combined_zone(unit_raster,lulc_raster,unit_lookup,unit_seq,id_field,name_field,lulc_labels,out_raster,out_lookup):
    rows=[]; zid=1
    for _,u in unit_lookup.iterrows():
        for lc,name in lulc_labels.items():
            rows.append({'zone_id':zid,unit_seq:int(u[unit_seq]),id_field:u[id_field],name_field:u[name_field],'lulc_code':int(lc),'lulc_name':name}); zid+=1
    lk=pd.DataFrame(rows); lk.to_csv(out_lookup,index=False)
    mapper={(r[unit_seq],r['lulc_code']):r['zone_id'] for _,r in lk.iterrows()}
    with rasterio.open(unit_raster) as us, rasterio.open(lulc_raster) as ls:
        prof=us.profile.copy(); prof.update(dtype='int32',nodata=0,compress='deflate')
        with rasterio.open(out_raster,'w',**prof) as dst:
            for _,w in us.block_windows(1):
                u=us.read(1,window=w); l=ls.read(1,window=w); out=np.zeros(u.shape,dtype=np.int32)
                for uu,ll in np.unique(np.column_stack((u.ravel(),l.ravel())),axis=0):
                    z=mapper.get((int(uu),int(ll)))
                    if z: out[(u==uu)&(l==ll)]=z
                dst.write(out,1,window=w)
    return lk


def pixel_area_ha(path):
    with rasterio.open(path) as s:
        if not s.crs or not s.crs.is_projected: raise ValueError('Expected projected equal-area raster')
        return abs(s.transform.a*s.transform.e)/10000.0


def qclass(n,area,cfg):
    if n>=cfg.PREFERRED_VALID_PIXELS and area>=cfg.PREFERRED_VALID_AREA_HA: return 'High'
    if n>=cfg.MIN_VALID_PIXELS and area>=cfg.MIN_VALID_AREA_HA: return 'Moderate'
    if n>=100: return 'Low'
    return 'Suppressed'


def categorical_summary(cat_path,zone_path,class_lookup,zone_lookup,zone_id,out_csv,cfg,valid_codes=None):
    pa=pixel_area_ha(cat_path); counts={}
    with rasterio.open(cat_path) as cs, rasterio.open(zone_path) as zs:
        for _,w in cs.block_windows(1):
            c=cs.read(1,window=w,masked=True); z=zs.read(1,window=w); valid=(~c.mask)&(z>0)
            if valid_codes is not None: valid &= np.isin(c.data,list(valid_codes))
            cc=c.data[valid].astype(np.int64); zz=z[valid].astype(np.int64)
            if not len(cc): continue
            m=max(class_lookup); enc=zz*(m+1)+cc
            for k,n in zip(*np.unique(enc,return_counts=True)):
                key=(int(k//(m+1)),int(k%(m+1))); counts[key]=counts.get(key,0)+int(n)
    totals={}
    for (z,_),n in counts.items(): totals[z]=totals.get(z,0)+n
    rows=[]
    for (z,c),n in counts.items():
        total=totals[z]; area=total*pa
        rows.append({zone_id:z,'class_code':c,'class_name':class_lookup.get(c,str(c)),'pixel_count':n,'area_ha':n*pa,'area_mha':n*pa/1e6,'share_pct':100*n/total,'valid_pixel_count':total,'valid_area_ha':area,'quality_class':qclass(total,area,cfg)})
    df=pd.DataFrame(rows)
    if not df.empty: df=df.merge(zone_lookup,on=zone_id,how='left')
    out_csv.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out_csv,index=False)
    try: df.to_parquet(out_csv.with_suffix('.parquet'),index=False)
    except Exception: pass
    return df


def crosstab(a_path,b_path,zone_path,lookup_a,lookup_b,zone_lookup,zone_id,out_csv,cfg):
    pa=pixel_area_ha(a_path); counts={}; ma=max(lookup_a); mb=max(lookup_b)
    with rasterio.open(a_path) as aa, rasterio.open(b_path) as bb, rasterio.open(zone_path) as zz:
        for _,w in aa.block_windows(1):
            a=aa.read(1,window=w,masked=True); b=bb.read(1,window=w,masked=True); z=zz.read(1,window=w)
            valid=(~a.mask)&(~b.mask)&(z>0)&np.isin(a.data,list(lookup_a))&np.isin(b.data,list(lookup_b))
            av=a.data[valid].astype(np.int64); bv=b.data[valid].astype(np.int64); zv=z[valid].astype(np.int64)
            if not len(av): continue
            enc=zv*((ma+1)*(mb+1))+av*(mb+1)+bv
            for k,n in zip(*np.unique(enc,return_counts=True)):
                zid=int(k//((ma+1)*(mb+1))); r=int(k%((ma+1)*(mb+1))); ca=int(r//(mb+1)); cb=int(r%(mb+1)); counts[(zid,ca,cb)]=counts.get((zid,ca,cb),0)+int(n)
    totals={}
    for (z,_,_),n in counts.items(): totals[z]=totals.get(z,0)+n
    rows=[]
    for (z,ca,cb),n in counts.items():
        t=totals[z]; area=t*pa; rows.append({zone_id:z,'class_a_code':ca,'class_a_name':lookup_a[ca],'class_b_code':cb,'class_b_name':lookup_b[cb],'pixel_count':n,'area_ha':n*pa,'area_mha':n*pa/1e6,'share_pct':100*n/t,'valid_pixel_count':t,'valid_area_ha':area,'quality_class':qclass(t,area,cfg)})
    df=pd.DataFrame(rows)
    if not df.empty: df=df.merge(zone_lookup,on=zone_id,how='left')
    df.to_csv(out_csv,index=False)
    try: df.to_parquet(out_csv.with_suffix('.parquet'),index=False)
    except Exception: pass
    return df


def agreement_metrics(cross,zone_id,class_codes,cfg):
    out=[]; idx={c:i for i,c in enumerate(class_codes)}
    for zid,g in cross.groupby(zone_id):
        cm=np.zeros((len(class_codes),len(class_codes)),dtype=int)
        for _,r in g.iterrows(): cm[idx[int(r.class_a_code)],idx[int(r.class_b_code)]]+=int(r.pixel_count)
        n=cm.sum(); rec={zone_id:zid,'valid_pixel_count':n,'exact_agreement':np.trace(cm)/n if n else np.nan}
        a=[];b=[]
        for i in range(len(class_codes)):
            for j in range(len(class_codes)):
                if cm[i,j]: a.extend([i]*cm[i,j]); b.extend([j]*cm[i,j])
        rec['weighted_kappa']=cohen_kappa_score(a,b,weights='quadratic') if n>=cfg.MIN_KAPPA_VALID_PIXELS and len(set(a))>1 and len(set(b))>1 else np.nan
        for code,i in idx.items():
            inter=cm[i,i]; union=cm[i,:].sum()+cm[:,i].sum()-inter
            rec[f'jaccard_class_{code}']=inter/union if union>=cfg.MIN_JACCARD_UNION_PIXELS and cm[i,:].sum()>=cfg.MIN_CLASS_PIXELS_EACH_MAP and cm[:,i].sum()>=cfg.MIN_CLASS_PIXELS_EACH_MAP else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def save_plot(fig,path,cfg):
    if cfg.SAVE_FIGURES:
        path.parent.mkdir(parents=True,exist_ok=True)
        for ext in cfg.FIGURE_FORMATS: fig.savefig(path.with_suffix('.'+ext),dpi=cfg.FIGURE_DPI,bbox_inches='tight')
    if cfg.SHOW_FIGURES: plt.show()
    else: plt.close(fig)


def stacked_bar(df,label,class_col,value,title,path,cfg):
    if df.empty:return
    units=df.groupby(label)[value].sum().sort_values(ascending=False).head(cfg.TOP_N_UNITS_IN_FIGURES).index
    p=df[df[label].isin(units)].pivot_table(index=label,columns=class_col,values=value,aggfunc='sum',fill_value=0)
    fig,ax=plt.subplots(figsize=(12,max(6,.35*len(p)))); p.plot(kind='barh',stacked=True,ax=ax); ax.set_title(title); ax.set_xlabel(value); ax.set_ylabel(''); ax.legend(title='',bbox_to_anchor=(1.02,1),loc='upper left'); fig.tight_layout(); save_plot(fig,path,cfg)
