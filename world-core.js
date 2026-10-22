// ModelImage v2 — inspectable moving visual atoms + self-discovered local relations.
// Only the synthetic world generator knows the eye/mouth rule.
// The learner sees complete RGB pixels and never receives semantic glyph names,
// absolute locations of the relevant parts, or a supplied product feature.
export const GRID=6, PATCH=16, WIDTH=GRID*PATCH, NSITES=GRID*GRID;
const OFFSETS=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
const clamp=(x,a=0,b=255)=>Math.max(a,Math.min(b,x));
export function rng(seed=1){let x=(seed>>>0)||1;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};}
function paintMotif(code,brightness=0) {
 const a=new Uint8ClampedArray(PATCH*PATCH*4);
 for(let y=0;y<PATCH;y++)for(let x=0;x<PATCH;x++){
  let r=112,g=141,b=166,dx=x-7.5,dy=y-7.5;
  if(code>=1&&code<=5){r=236;g=216;b=177;}
  if(code===1 && dx*dx+dy*dy<16){r=38;g=49;b=60;} // round eye
  if(code===2 && Math.abs(dy)<2 && Math.abs(dx)<5){r=38;g=49;b=60;} // flat eye
  if(code===3 && Math.abs(dy-(2.4-0.077*dx*dx))<1.3 && Math.abs(dx)<6){r=112;g=53;b=66;} // smile
  if(code===4 && Math.abs(dy-(-2.4+0.077*dx*dx))<1.3 && Math.abs(dx)<6){r=112;g=53;b=66;} // frown
  if(code===5 && (dx*dx+dy*dy<6 || (x>2&&x<5&&y>8&&y<11))){r=178;g=110;b=110;} // cheek
  if(code===6 && (Math.abs(dx)<2 || Math.abs(dy)<2)){r=109;g=210;b=176;} // distractor plus
  if(code===7 && (Math.abs(dx+dy)<2 || Math.abs(dx-dy)<2)){r=232;g=193;b=87;} // distractor cross
  if(code===8 && dx*dx+dy*dy<20){r=167;g=138;b=224;} // distractor dot
  const k=(y*PATCH+x)*4;
  a[k]=clamp(r+brightness);a[k+1]=clamp(g+brightness);a[k+2]=clamp(b+brightness);a[k+3]=255;
 }
 return a;
}
export function imageFromPatches(patches){
 const out=new Uint8ClampedArray(WIDTH*WIDTH*4);
 for(let p=0;p<NSITES;p++){
  const gx=p%GRID,gy=Math.floor(p/GRID),t=patches[p];
  for(let y=0;y<PATCH;y++){
   const dst=((gy*PATCH+y)*WIDTH+gx*PATCH)*4;
   out.set(t.subarray(y*PATCH*4,(y+1)*PATCH*4),dst);
  }
 }
 return {width:WIDTH,height:WIDTH,data:out};
}
export function makeWorld(seed,world="parity",distractors=8){
 const rand=rng(seed), codes=Array(NSITES).fill(0);
 const fx=1+Math.floor(rand()*4),fy=Math.floor(rand()*5);
 const e1=rand()<.5?1:2,e2=rand()<.5?1:2;
 let mouth=rand()<.5?3:4;
 if(world==="parity")mouth=(e1===e2)?3:4;
 if(world==="opposite")mouth=(e1===e2)?4:3;
 if(world==="copy")mouth=e1===1?3:4;
 const p=fy*GRID+fx;
 codes[p]=e1;codes[p+1]=e2;codes[p+GRID]=mouth;codes[p+GRID+1]=5;
 const blocked=new Set([p,p+1,p+GRID,p+GRID+1]);let placed=0,attempts=0;
 while(placed<distractors&&attempts++<300){
  const i=Math.floor(rand()*NSITES);if(blocked.has(i)||codes[i]!==0)continue;
  codes[i]=6+Math.floor(rand()*3);placed++;
 }
 const brightness=(rand()-.5)*3;
 const patches=codes.map(c=>paintMotif(c,brightness));
 return {...imageFromPatches(patches),face:{x:fx,y:fy,eyeLeft:p,eyeRight:p+1,mouth:p+GRID},codes};
}
export function imagePatches(img){
 if(img.width!==WIDTH||img.height!==WIDTH)throw Error("96×96 RGBA image required");
 const a=[];
 for(let p=0;p<NSITES;p++){
  const tile=new Uint8ClampedArray(PATCH*PATCH*4),sx=(p%GRID)*PATCH,sy=Math.floor(p/GRID)*PATCH;
  for(let y=0;y<PATCH;y++){
   const start=((sy+y)*WIDTH+sx)*4;
   tile.set(img.data.subarray(start,start+PATCH*4),y*PATCH*4);
  }
  a.push(tile);
 }
 return a;
}
function features(tile){
 const f=new Float32Array(4*4*3),block=PATCH/4;
 for(let y=0;y<PATCH;y++)for(let x=0;x<PATCH;x++){
  const src=(y*PATCH+x)*4,dst=(Math.floor(y/block)*4+Math.floor(x/block))*3;
  for(let c=0;c<3;c++)f[dst+c]+=tile[src+c]/(255*block*block);
 }
 return f;
}
function fd(a,b){let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d;}return s/a.length;}
function neighbors(site) {
 const x=site%GRID,y=Math.floor(site/GRID);
 return OFFSETS.map(([dx,dy])=>{const xx=x+dx,yy=y+dy;return xx>=0&&yy>=0&&xx<GRID&&yy<GRID?yy*GRID+xx:-1;});
}
function increment(map,key,target){
 let value=map.get(key);
 if(!value){value={n:0,counts:new Map()};map.set(key,value);}
 value.n++;value.counts.set(target,(value.counts.get(target)||0)+1);
}
export class MovingSieve {
 constructor({radius=.003,maxAtoms=24}={}){
  this.radius=radius;this.maxAtoms=maxAtoms;this.atoms=[];
  this.base=new Map();this.unary=new Map();this.binary=new Map();this.lessons=0;
 }
 observe(tile,train=false){
  const f=features(tile);let nearest=-1,ds=Infinity;
  for(let i=0;i<this.atoms.length;i++){
   const dist=fd(f,this.atoms[i].f);
   if(dist<ds){ds=dist;nearest=i;}
  }
  if(train&&(nearest<0||ds>this.radius)&&this.atoms.length<this.maxAtoms){
   nearest=this.atoms.length;
   this.atoms.push({f:Array.from(f),pixels:Array.from(tile),count:1});
   return nearest;
  }
  if(train&&nearest>=0){
   const a=this.atoms[nearest],n=++a.count,eta=1/n;
   for(let k=0;k<f.length;k++)a.f[k]+=(f[k]-a.f[k])*eta;
   for(let k=0;k<tile.length;k++)a.pixels[k]+=(tile[k]-a.pixels[k])*eta;
  }
  return nearest;
 }
 teach(img){
  const tiles=imagePatches(img);
  // Learn dictionary first; learn relations from the resulting **observed** patches.
  const codes=tiles.map(t=>this.observe(t,true));
  // Re-encode after the dictionary has been extended by the whole scene.
  for(let p=0;p<NSITES;p++)codes[p]=this.observe(tiles[p],false);
  for(let p=0;p<NSITES;p++){
   const t=codes[p];if(t<0)continue;
   this.base.set(t,(this.base.get(t)||0)+1);
   const near=neighbors(p);
   for(let i=0;i<OFFSETS.length;i++){
    const a=near[i];if(a<0||codes[a]<0)continue;
    increment(this.unary,i+"|"+codes[a],t);
    for(let j=i+1;j<OFFSETS.length;j++){
     const b=near[j];if(b<0||codes[b]<0)continue;
     increment(this.binary,i+","+j+"|"+codes[a]+","+codes[b],t);
    }
   }
  }
  this.lessons++;return codes;
 }
 encode(img){return imagePatches(img).map(t=>this.observe(t,false));}
 prior(code){
  const total=this.lessons*NSITES;
  return ((this.base.get(code)||0)+.1)/(total+.1*Math.max(1,this.atoms.length));
 }
 // All candidate offsets are scanned. No relative geometry, symbol ID, or three-part formula is supplied.
 candidates(codes,site,order=2){
  const near=neighbors(site),map=order===2?this.binary:this.unary;
  let best=null;
  const probe=(key,positions)=>{
   const rec=map.get(key);if(!rec||rec.n<8)return;
   for(const [target,count] of rec.counts){
    if(count<5)continue;
    const prior=this.prior(target);
    const conditional=(count+.1)/(rec.n+.1*this.atoms.length);
    const lift=Math.log(Math.max(1e-12,conditional/prior));
    const specificity=1-Math.exp(-rec.n/30);
    // Preference for a repeatedly verified relation over chance rare coincidences.
    const score=lift*specificity;
    if(score>0&&(!best||score>best.score)){
     best={code:target,score,confidence:conditional,count,support:rec.n,positions,order};
    }
   }
  };
  for(let i=0;i<near.length;i++){
   const a=near[i];if(a<0||codes[a]<0)continue;
   if(order===1)probe(i+"|"+codes[a],[a]);
   else for(let j=i+1;j<near.length;j++){
    const b=near[j];if(b<0||codes[b]<0)continue;
    probe(i+","+j+"|"+codes[a]+","+codes[b],[a,b]);
   }
  }
  return best;
 }
 predict(codes,site,order=2){
  const best=this.candidates(codes,site,order);
  if(best)return best;
  let frequent=-1,mx=0;
  for(const [code,n] of this.base)if(n>mx){mx=n;frequent=code;}
  return {code:frequent,score:0,confidence:this.prior(frequent),count:0,support:0,positions:[],order:0};
 }
 complete(img,hidden,order=2,passes=3){
  const observed=this.encode(img),codes=observed.slice();
  hidden.forEach(p=>{codes[p]=-1;});
  const provenance=Array(NSITES).fill(null);
  // Update strongest explanations first to avoid making future predictions from low-confidence guesses.
  for(let turn=0;turn<passes;turn++){
   const options=[];
   for(const p of hidden){
    if(codes[p]>=0)continue;
    const candidate=this.predict(codes,p,order);
    if(candidate.code>=0)options.push({p,...candidate});
   }
   options.sort((a,b)=>b.score-a.score);
   if(!options.length)break;
   const high=options.filter(c=>c.score>.36);
   const selected=high.length?high:turn===passes-1?options:[];
   if(!selected.length)continue;
   for(const c of selected){codes[c.p]=c.code;provenance[c.p]=c;}
   // All high-evidence predictions from this pass are committed simultaneously.
  }
  for(const p of hidden)if(codes[p]<0){const c=this.predict(codes,p,order);codes[p]=c.code;provenance[p]=c;}
  const generated=this.render(codes);
  // Never replace visible sensor pixels with blurry codebook reconstructions.
  const pixels=new Uint8ClampedArray(img.data);
  for(const p of hidden){
   const sx=(p%GRID)*PATCH,sy=Math.floor(p/GRID)*PATCH;
   for(let y=0;y<PATCH;y++){
    const start=((sy+y)*WIDTH+sx)*4;
    pixels.set(generated.data.subarray(start,start+PATCH*4),start);
   }
  }
  return {codes,provenance,image:{width:WIDTH,height:WIDTH,data:pixels},observed};
 }
 render(codes){
  const parts=codes.map(c=>{
   const t=this.atoms[c]?.pixels;
   if(!t)return paintMotif(0);
   return Uint8ClampedArray.from(t);
  });
  return imageFromPatches(parts);
 }
}
export function halfMask(seed=100){
 const rand=rng(seed),hidden=new Set();
 // A contiguous 3-column occlusion: exposes how much context disappears.
 const side=rand()<.5?0:3;
 for(let y=0;y<GRID;y++)for(let x=side;x<side+3;x++)hidden.add(y*GRID+x);
 return [...hidden];
}
export function maskedError(pred,truth,hidden){
 let d=0;
 for(const p of hidden){
  const x=(p%GRID)*PATCH,y=Math.floor(p/GRID)*PATCH;
  for(let iy=0;iy<PATCH;iy++)for(let ix=0;ix<PATCH;ix++){
   const k=((y+iy)*WIDTH+x+ix)*4;
   for(let c=0;c<3;c++){const z=(pred.data[k+c]-truth.data[k+c])/255;d+=z*z;}
  }
 }
 return d/(hidden.length*PATCH*PATCH*3);
}
export function makeMasked(img,hidden){
 const a=new Uint8ClampedArray(img.data);
 for(const p of hidden){
  const x=(p%GRID)*PATCH,y=Math.floor(p/GRID)*PATCH;
  for(let iy=0;iy<PATCH;iy++)for(let ix=0;ix<PATCH;ix++){
   const k=((y+iy)*WIDTH+x+ix)*4;
   a[k]=44;a[k+1]=53;a[k+2]=61;a[k+3]=255;
  }
 }
 return {width:WIDTH,height:WIDTH,data:a};
}
