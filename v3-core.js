// ModelImage V3 — hierarchical residue dictionary and adaptive predictive groups.
// Standalone ES module. No pretrained weights, semantic labels or external assets.
// All prediction APIs accept a mask and encode only OBSERVED pixels.
export const N=96, P=8, G=N/P, CELLS=G*G;
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
export const patchIndex=(x,y)=>y*G+x;
export function random(seed=1){let x=seed>>>0||1;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};}
export function cropRGBA(img){
 if(img.width!==N||img.height!==N)throw Error('Expected 96×96 RGBA image');
 return img;
}
export function patchPixels(img,p){
 const d=new Float32Array(P*P*3),x=(p%G)*P,y=Math.floor(p/G)*P;
 for(let j=0;j<P;j++)for(let i=0;i<P;i++){
  const at=((y+j)*N+x+i)*4,to=(j*P+i)*3;
  for(let c=0;c<3;c++)d[to+c]=img.data[at+c]/255;
 }
 return d;
}
function avg2(pixels,size){const step=P/size,z=new Float32Array(size*size*3);
 for(let y=0;y<P;y++)for(let x=0;x<P;x++){
  const i=(Math.floor(y/step)*size+Math.floor(x/step))*3,j=(y*P+x)*3;
  for(let c=0;c<3;c++)z[i+c]+=pixels[j+c]/(step*step);
 }return z;
}
function expand(values,size,dest=8){
 const data=new Float32Array(dest*dest*3);
 for(let y=0;y<dest;y++)for(let x=0;x<dest;x++){
  // Piecewise-linear interpolation in color-space; true progressive spatial scales.
  const fx=clamp((x+.5)*size/dest-.5,0,size-1),fy=clamp((y+.5)*size/dest-.5,0,size-1);
  const ix=Math.floor(fx),iy=Math.floor(fy),jx=Math.min(size-1,ix+1),jy=Math.min(size-1,iy+1);
  const dx=fx-ix,dy=fy-iy,to=(y*dest+x)*3;
  for(let c=0;c<3;c++){
   const a=values[(iy*size+ix)*3+c],b=values[(iy*size+jx)*3+c];
   const e=values[(jy*size+ix)*3+c],f=values[(jy*size+jx)*3+c];
   data[to+c]=(1-dy)*((1-dx)*a+dx*b)+dy*((1-dx)*e+dx*f);
  }
 }return data;
}
export function sieve(pixels){
 // Explicit additive hierarchy: coarse + mid residue + fine residue = original (within roundoff).
 const coarse=avg2(pixels,2),midPixel=avg2(pixels,4);
 const midBase=expand(coarse,2,4),mid=new Float32Array(4*4*3);
 for(let i=0;i<mid.length;i++)mid[i]=midPixel[i]-midBase[i];
 const low4=expand(midPixel,4,8),fine=new Float32Array(P*P*3);
 for(let i=0;i<fine.length;i++)fine[i]=pixels[i]-low4[i];
 return [coarse,mid,fine];
}
export function synthesize(coarse,mid,fine,detail=1){
 const mbase=expand(coarse,2,4),m=new Float32Array(48);
 for(let i=0;i<m.length;i++)m[i]=mbase[i]+(mid?mid[i]:0)*detail;
 const out=expand(m,4,8);
 for(let i=0;i<out.length;i++)out[i]=clamp(out[i]+(fine?fine[i]:0)*detail);
 return out;
}
function distance(a,b){let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d;}return s/a.length;}
class CompetitiveDictionary{
 constructor(dim,cap,threshold,rate){this.dim=dim;this.cap=cap;this.threshold=threshold;this.rate=rate;this.atoms=[];}
 winner(x,learn=false){
  let win=-1,dist=Infinity;
  for(let i=0;i<this.atoms.length;i++){
   const e=distance(x,this.atoms[i].w);
   if(e<dist){dist=e;win=i;}
  }
  if(learn&&(win<0||dist>this.threshold)&&this.atoms.length<this.cap){
   win=this.atoms.length;this.atoms.push({w:Array.from(x),hits:1});return win;
  }
  if(learn&&win>=0){
   const a=this.atoms[win],eta=this.rate/Math.sqrt(1+a.hits/8);
   for(let i=0;i<x.length;i++)a.w[i]+=eta*(x[i]-a.w[i]);
   a.hits++;
  }
  return win;
 }
 get(id){return this.atoms[id]?.w||null;}
}
// 8 local offsets plus 4 longer cross-links. Shared over every grid translation.
export const OFFSETS=[[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,-1],[-1,1],[1,1],[-2,0],[2,0],[0,-2],[0,2]];
function atOffset(p,d){
 const x=p%G+OFFSETS[d][0],y=Math.floor(p/G)+OFFSETS[d][1];
 return x>=0&&y>=0&&x<G&&y<G?patchIndex(x,y):-1;
}
class SparseRelations{
 constructor(){this.one=new Map();this.two=new Map();this.freq=new Map();this.total=0;}
 add(map,key,val){
  let r=map.get(key);if(!r){r={n:0,targets:new Map()};map.set(key,r);}
  r.n++;r.targets.set(val,(r.targets.get(val)||0)+1);
 }
 teach(codes,rng){
  for(let p=0;p<CELLS;p++){
   const target=codes[p];if(target<0)continue;
   this.total++;this.freq.set(target,(this.freq.get(target)||0)+1);
   const avail=[];
   for(let i=0;i<OFFSETS.length;i++){
    const q=atOffset(p,i);if(q<0||codes[q]<0)continue;
    avail.push([i,codes[q]]);
    this.add(this.one,i+'|'+codes[q],target);
   }
   // Learn a budgeted rotating sample of conjunctions: prevents O(image * 66 * lessons) blowup.
   const nPairs=avail.length*(avail.length-1)/2;
   const budget=Math.min(nPairs,8);
   if(nPairs===0)continue;
   const start=Math.floor(rng()*nPairs);
   const jump=13; // relatively prime to common candidate-pair counts; no privileged offsets
   for(let k=0;k<budget;k++){
    const idx=(start+k*jump)%nPairs;
    let at=0,pair=null;
    for(let i=0;i<avail.length&&!pair;i++)for(let j=i+1;j<avail.length;j++){
     if(at++===idx){pair=[avail[i],avail[j]];break;}
    }
    if(pair){
     const [[i,a],[j,b]]=pair;
     this.add(this.two,i+','+j+'|'+a+','+b,target);
    }
   }
  }
 }
 lookup(map,key,order,positions){
  const record=map.get(key);if(!record||record.n<4)return null;
  let best=null;
  for(const [code,n] of record.targets){
   if(n<2)continue;
   const conditional=(n+.3)/(record.n+.3*Math.max(1,this.freq.size));
   const prior=(this.freq.get(code)+.3)/(this.total+.3*Math.max(1,this.freq.size));
   const lift=Math.log(Math.max(1e-8,conditional/prior));
   const confidence=1-Math.exp(-record.n/15);
   const score=lift*confidence+(order===2?.06:0);
   if(score>.1&&(!best||score>best.score))best={code,score,prob:conditional,observations:n,support:record.n,positions,order};
  }
  return best;
 }
 predict(codes,p,order=2){
  const avail=[];
  for(let i=0;i<OFFSETS.length;i++){
   const q=atOffset(p,i);if(q>=0&&codes[q]>=0)avail.push([i,codes[q],q]);
  }
  let best=null;
  for(let i=0;i<avail.length;i++){
   const [d,a,site]=avail[i];
   const c=this.lookup(this.one,d+'|'+a,1,[site]);
   if(c&&(!best||c.score>best.score))best=c;
   if(order===1)continue;
   for(let j=i+1;j<avail.length;j++){
    const [e,b,other]=avail[j];
    const z=this.lookup(this.two,d+','+e+'|'+a+','+b,2,[site,other]);
    if(z&&(!best||z.score>best.score))best=z;
   }
  }
  if(best)return best;
  let code=-1,n=-1;
  for(const [k,v] of this.freq)if(v>n){code=k;n=v;}
  return {code,score:0,prob:this.total?n/this.total:0,observations:0,support:0,positions:[],order:0};
 }
}
export class ResidueWorld{
 constructor({caps=[48,64,80],thresholds=[.007,.003,.002],rate=.19,seed=17}={}){
  this.bands=[new CompetitiveDictionary(12,caps[0],thresholds[0],rate),
   new CompetitiveDictionary(48,caps[1],thresholds[1],rate),
   new CompetitiveDictionary(192,caps[2],thresholds[2],rate)];
  this.groups=new SparseRelations();this.middle=new Map();this.detail=new Map();
  this.lessons=0;this.examples=0;this.rng=random(seed);
 }
 addConditional(map,key,value){
  let v=map.get(key);if(!v){v=new Map();map.set(key,v);}
  v.set(value,(v.get(value)||0)+1);
 }
 teach(image){
  cropRGBA(image);const coarseCodes=[],middles=[],fines=[];
  for(let p=0;p<CELLS;p++){
   const [c,m,f]=sieve(patchPixels(image,p));
   const a=this.bands[0].winner(c,true),b=this.bands[1].winner(m,true),d=this.bands[2].winner(f,true);
   coarseCodes.push(a);middles.push(b);fines.push(d);
   this.addConditional(this.middle,a,b);
   this.addConditional(this.detail,a+'|'+b,d);
  }
  this.groups.teach(coarseCodes,this.rng);
  this.examples+=CELLS;this.lessons++;return this;
 }
 recognize(image,mask=[]){
  cropRGBA(image);const hidden=new Set(mask),coarse=Array(CELLS).fill(-1),mid=Array(CELLS).fill(-1),fine=Array(CELLS).fill(-1);
  for(let p=0;p<CELLS;p++)if(!hidden.has(p)){
   const [c,m,f]=sieve(patchPixels(image,p));
   coarse[p]=this.bands[0].winner(c,false);
   mid[p]=this.bands[1].winner(m,false);
   fine[p]=this.bands[2].winner(f,false);
  }
  return {coarse,mid,fine};
 }
 choose(map,key){
  const counts=map.get(key);if(!counts)return -1;
  let best=-1,n=-1;for(const [k,v] of counts){if(v>n){n=v;best=k;}}
  return best;
 }
 // Proposal from learned atoms + adaptively selected, movable 1/2 context groups.
 // There is NO reference to target pixel content when the target is masked.
 complete(image,hidden,{order=2,detail=1,passes=3}={}){
  const mask=new Set(hidden),obs=this.recognize(image,hidden);
  const codes=obs.coarse.slice(),provenance=Array(CELLS).fill(null);
  for(let turn=0;turn<passes;turn++){
   const proposals=[];
   for(const p of mask)if(codes[p]<0){
    const c=this.groups.predict(codes,p,order);
    if(c.code>=0)proposals.push({p,...c});
   }
   proposals.sort((a,b)=>b.score-a.score);
   // Prefer high-certainty groups. Last turn backfills uncertain sites.
   const picks=turn===passes-1?proposals:proposals.filter(x=>x.score>.4);
   for(const x of picks){codes[x.p]=x.code;provenance[x.p]=x;}
  }
  const output=new Uint8ClampedArray(image.data);
  const recon=[0,0,0],bandCount=this.bands.map(b=>b.atoms.length);
  for(const p of mask){
   const a=codes[p],b=this.choose(this.middle,a),d=this.choose(this.detail,a+'|'+b);
   const coarse=this.bands[0].get(a)||Array(12).fill(.5);
   const mid=this.bands[1].get(b)||Array(48).fill(0),fine=this.bands[2].get(d)||Array(192).fill(0);
   const full=synthesize(coarse,mid,fine,detail),x=(p%G)*P,y=Math.floor(p/G)*P;
   for(let iy=0;iy<P;iy++)for(let ix=0;ix<P;ix++){
    const offset=((y+iy)*N+x+ix)*4,src=(iy*P+ix)*3;
    for(let col=0;col<3;col++)output[offset+col]=Math.round(full[src+col]*255);
    output[offset+3]=255;
   }
   if(a>=0)recon[0]++;if(b>=0)recon[1]++;if(d>=0)recon[2]++;
  }
  return {image:{width:N,height:N,data:output},codes,provenance,bandCount,coverage:recon.map(n=>mask.size?n/mask.size:0)};
 }
 summary(){
  return {lessons:this.lessons,atoms:this.bands.map(b=>b.atoms.length),
   spatialPairs:this.groups.one.size,spatialTriples:this.groups.two.size,
   coarseToMid:this.middle.size,midToFine:this.detail.size};
 }
 // Model persistence: full, inspectable JSON; never saves training photographs.
 export(){return {version:3,lessons:this.lessons,examples:this.examples,atoms:this.bands.map(b=>b.atoms),
  freq:[...this.groups.freq],total:this.groups.total,
  one:[...this.groups.one].map(([k,v])=>[k,{n:v.n,targets:[...v.targets]}]),
  two:[...this.groups.two].map(([k,v])=>[k,{n:v.n,targets:[...v.targets]}]),
  middle:[...this.middle].map(([k,v])=>[k,[...v]]),detail:[...this.detail].map(([k,v])=>[k,[...v]])};}
}
export function maskRectangle(x,y,w,h){
 const a=[];for(let iy=y;iy<y+h;iy++)for(let ix=x;ix<x+w;ix++)if(ix>=0&&iy>=0&&ix<G&&iy<G)a.push(patchIndex(ix,iy));return a;
}
export function maskedImage(image,mask){
 const d=new Uint8ClampedArray(image.data);
 for(const p of mask){
  const x=(p%G)*P,y=Math.floor(p/G)*P;
  for(let iy=0;iy<P;iy++)for(let ix=0;ix<P;ix++){
   const k=((y+iy)*N+x+ix)*4;
   d[k]=49;d[k+1]=59;d[k+2]=68;d[k+3]=255;
  }
 }return {width:N,height:N,data:d};
}
export function maskedMSE(image,truth,mask){
 if(!mask.length)return 0;
 let s=0;for(const p of mask){
  const x=p%G*P,y=Math.floor(p/G)*P;
  for(let iy=0;iy<P;iy++)for(let ix=0;ix<P;ix++){
   const k=((y+iy)*N+x+ix)*4;
   for(let c=0;c<3;c++){const d=(image.data[k+c]-truth.data[k+c])/255;s+=d*d;}
  }
 }return s/(mask.length*P*P*3);
}
export function naiveCompletion(img,mask){
 // Nonlearned four-neighbor spatial fallback used only as benchmark.
 const hidden=new Set(mask),out=new Uint8ClampedArray(img.data);
 for(const p of mask){
  const x=p%G,y=Math.floor(p/G),available=[];
  for(let j=0;j<CELLS;j++)if(!hidden.has(j)){
   const dx=j%G-x,dy=Math.floor(j/G)-y;
   const dist=dx*dx+dy*dy;if(dist<18)available.push([j,dist]);
  }
  available.sort((a,b)=>a[1]-b[1]);
  const picks=available.slice(0,5),sums=[0,0,0],wSum=picks.reduce((a,b)=>a+1/(b[1]+.5),0)||1;
  for(const [q,dist] of picks){
   const px=q%G*P,py=Math.floor(q/G)*P,w=1/(dist+.5);
   for(let iy=0;iy<P;iy++)for(let ix=0;ix<P;ix++){
    const k=((py+iy)*N+px+ix)*4;
    for(let c=0;c<3;c++)sums[c]+=img.data[k+c]*w/(P*P);
   }
  }
  for(let iy=0;iy<P;iy++)for(let ix=0;ix<P;ix++){
   const k=(((y*P)+iy)*N+x*P+ix)*4;
   for(let c=0;c<3;c++)out[k+c]=Math.round(sums[c]/wSum);out[k+3]=255;
  }
 }
 return {width:N,height:N,data:out};
}
