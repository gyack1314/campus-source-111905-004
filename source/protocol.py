"""按变量维度索引读取的 NetCDF 合成阵列服务。"""
import io
import json
import numpy as np
from scipy.io import netcdf_file

def js(v):return 200,{"Content-Type":"application/json"},json.dumps(v,ensure_ascii=False).encode()
class Source:
    name="教学实验阵列"
    def catalog(self,version):
        return dict(version=version,max_cells=4,
          axes=dict(time=[0,1,2,3,4,5] if version=="r1" else [0,1,2,2.5,3,4,5],
                    station=[30,20,10],sample=[0,.4,1.8,4.6]),
          variables=dict(temperature=dict(dimensions=["time","station"],scale_factor=.1,add_offset=273.15,fill=-9999),
                         current=dict(dimensions=["station","sample"],scale_factor=.01,add_offset=0,fill=-9999)))
    def route(self,method,path,q,headers,body):
        version=q.get("version","r2" if self.stage>=1 else "r1")
        if version not in ["r1","r2"]:return 404,{},b""
        if version=="r1" and self.stage>=2:return 410,{"X-Source-Version":version},"发布版租期已结束"
        cat=self.catalog(version)
        if path=="/catalog":return js(cat)
        if path!="/array.nc":return 404,{},b""
        variable=q["variable"];meta=cat["variables"][variable];dims=meta["dimensions"];indices=[]
        for dim in dims:
            axis=cat["axes"][dim];start,stop=map(int,q.get(dim,f"0:{len(axis)}").split(":"))
            if not 0<=start<=stop<=len(axis):raise ValueError("索引越界")
            indices.append(list(range(start,stop)))
        count=int(np.prod([len(a) for a in indices]))
        if count>cat["max_cells"]:return 413,{},"单次单元数超限"
        buffer=io.BytesIO();f=netcdf_file(buffer,"w")
        f.source_version=version
        # NetCDF classic 的零长维度只有一个无限维度，因此空选择返回显式空响应。
        if count==0:return 204,{"X-Source-Version":version},b""
        for dim,idx in zip(dims,indices):
            f.createDimension(dim,len(idx))
            a=f.createVariable(dim,"f8",(dim,));a[:]=np.array(cat["axes"][dim])[idx]
        values=f.createVariable(variable,"i2",tuple(dims))
        quality=f.createVariable("quality","i1",tuple(dims))
        values.scale_factor=meta["scale_factor"];values.add_offset=meta["add_offset"];values._FillValue=np.int16(-9999)
        raw=np.empty(tuple(len(x) for x in indices),dtype=np.int16)
        mask=np.ones(raw.shape,dtype=np.int8)
        for local in np.ndindex(raw.shape):
            absolute=tuple(indices[k][pos] for k,pos in enumerate(local))
            raw[local]=100+absolute[0]*10+absolute[1]
            if absolute==(1,1):raw[local]=-9999;mask[local]=0
            if version=="r2" and absolute==(2,0):mask[local]=0
        values[:]=raw;quality[:]=mask
        f.flush();data=buffer.getvalue();f.close()
        return 200,{"Content-Type":"application/x-netcdf","X-Source-Version":version},data
