// ModelImage: readable, trainable tile/relationship assembly model.
// Pure JavaScript, no dependencies; runs in browsers and Node 22+.
// The target index is used ONLY for the displayed evaluation metric.
export const GRID=6, SIDE=96, TILE=16, COUNT=GRID*GRID;
export function random(seed=1){let s=(seed>>>0)||1;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)/4294967296;};}
const clamp=(x,a=0,b=255)=>Math.max(a,Math.min(b,x));
const sq=x=>x*x;
function blob(x,y,cx,cy,rx,ry){return sq((x-cx)/rx)+sq((y-cy)/ry)<1;}
export function synthetic(theme="face",seed=1){
 const rand=random(seed), w=SIDE, h=SIDE, d=new Uint8ClampedArray(w*h*4);
 const j=(rand()-.5)*.035, k=(rand()-.5)*.035, shade=(rand()-.5)*26;
 const lights=Array.from({length:4},()=>({x:rand(),y:rand(),r:.035+rand()*.03}));
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  const u=x/w,v=y/h; let r=0,g=0,b=0;
  if(theme==="face"){
   r=69+shade*.2;g=98+shade*.2;b=129+shade*.2;
   if(blob(u,v,.5+j,.52+k,.315,.425)){r=211+shade;g=159+shade*.6;b=116+shade*.4;}
   if(blob(u,v,.361+j,.448+k,.051,.046)||blob(u,v,.639+j,.448+k,.051,.046)){r=246;g=237;b=219;}
   if(blob(u,v,.365+j,.45+k,.024,.033)||blob(u,v,.635+j,.45+k,.024,.033)){r=43;g=60;b=73;}
   if(blob(u,v,.5+j,.565+k,.030,.067)){r=183;g=118;b=89;}
   const yy=.706+k+.035*Math.cos((u-.5)*10),dist=Math.abs(v-yy);
   if(u>.36+j&&u<.64+j&&dist<.009){r=111;g=56;b=70;}
   if((Math.abs(v-(.367+k))<.014)&&(Math.abs(u-(.36+j))<.073||Math.abs(u-(.64+j))<.073)){r=88;g=65;b=60;}
  }else if(theme==="garden"){
   const hill=.55+.03*Math.sin(13*u+seed*.1);
   if(v<hill){r=119+v*74+shade*.2;g=186+v*35;b=223-v*15;}
   else{r=75-v*20;g=138+v*38+shade*.2;b=76;}
   for(let t=0;t<4;t++){
    const cx=(t+1)/5+.018*Math.sin(seed*.41+t), cy=.39+.03*Math.cos(seed*.71+t);
    if(v>cy&&v<.89&&Math.abs(u-cx)<.009){r=40;g=113;b=52;}
    const dx=u-cx,dy=v-cy,ang=Math.atan2(dy,dx),rho=Math.hypot(dx,dy);
    if(rho<.024||Math.abs(rho-(.058+.010*Math.cos(6*ang)))<.016&&rho<.082){
      if(rho<.027){r=234;g=153;b=52;}else{r=230-(t%2)*40;g=100+32*(t%2);b=139+30*(t%2);}
    }
   }
  }else if(theme==="city"){
   r=108+64*v+shade*.2;g=161+46*v;b=212-33*v;
   if(v>.79){r=74;g=87;b=91;}
   for(let t=0;t<5;t++){
    const a=t*.21-.01, right=a+.18, top=.34+.11*Math.sin(2.3*t+seed*.24);
    if(u>a&&u<right&&v>top&&v<.80){r=64+t*17;g=73+t*11;b=93+t*8;
      const ix=Math.floor((u-a)/.047),iy=Math.floor((v-top)/.062);
      if((u-a)% .047>.009&&(u-a)% .047<.035&&(v-top)% .062>.014&&(v-top)% .062<.048){
       if((ix+iy+t+seed)%4!==0){r=228;g=183;b=107;}
      }
    }
   }
  }else{
   let z=.48+.22*Math.sin(13*u+4*Math.sin(6*v+j*10))+.15*Math.cos(19*v+7*u);
   for(const l of lights)if(blob(u,v,l.x,l.y,l.r,l.r)){z+=.3;}
   r=90+140*z;g=80+130*(1-z);b=135+90*Math.sin(3*z);
  }
  const n=(rand()-.5)*7,at=(y*w+x)*4;
  d[at]=clamp(r+n);d[at+1]=clamp(g+n);d[at+2]=clamp(b+n);d[at+3]=255;
 }
 return {width:w,height:h,data:d};
}
function feature(px){
 const z=new Float32Array(15), sums=new Float32Array(9);let lum2=0;
 for(let y=0;y<TILE;y++)for(let x=0;x<TILE;x++){
  const p=(y*TILE+x)*4,r=px[p]/255,g=px[p+1]/255,b=px[p+2]/255;
  const l=.299*r+.587*g+.114*b;
  z[0]+=r;z[1]+=g;z[2]+=b;
  sums[Math.floor(y*3/TILE)*3+Math.floor(x*3/TILE)]+=l;
  z[12]+=l*(x-(TILE-1)/2);z[13]+=l*(y-(TILE-1)/2);lum2+=l*l;
 }
 const area=TILE*TILE;for(let i=0;i<3;i++)z[i]/=area;
 const cellN=[36,30,30,30,25,25,30,25,25]; // overwritten by actual cell sizes below
 for(let a=0;a<3;a++)for(let b=0;b<3;b++){
  let c=0;for(let y=0;y<TILE;y++)for(let x=0;x<TILE;x++)if(Math.floor(y*3/TILE)===a&&Math.floor(x*3/TILE)===b)c++;
  z[3+a*3+b]=sums[a*3+b]/c;
 }
 z[12]/=area*TILE/4;z[13]/=area*TILE/4;
 const mean=.299*z[0]+.587*z[1]+.114*z[2];
 z[14]=Math.sqrt(Math.max(0,lum2/area-mean*mean));
 return z;
}
export function tiles(image){
 if(image.width!==SIDE||image.height!==SIDE)throw Error("Expected 96x96 RGBA image");
 const out=[];
 for(let gy=0;gy<GRID;gy++)for(let gx=0;gx<GRID;gx++){
  const pix=new Uint8ClampedArray(TILE*TILE*4);
  for(let y=0;y<TILE;y++){
   const offset=((gy*TILE+y)*SIDE+gx*TILE)*4;
   pix.set(image.data.subarray(offset,offset+TILE*4),y*TILE*4);
  }
  out.push({pix,feature:feature(pix),truth:gy*GRID+gx});
 }
 return out;
}
export function emptyModel(){return {count:0,slots:Array.from({length:COUNT},()=>Array(15).fill(0)),links:Array.from({length:COUNT*2},()=>Array(15).fill(0)),lastTheme:"none"};}
export function teach(model,image){
 const parts=tiles(image),n=model.count+1;
 for(let p=0;p<COUNT;p++){
  for(let f=0;f<15;f++)model.slots[p][f]+=(parts[p].feature[f]-model.slots[p][f])/n;
  for(let dir=0;dir<2;dir++){
   const q=dir===0?(p%GRID===GRID-1?-1:p+1):(p>=COUNT-GRID?-1:p+GRID);
   if(q<0)continue;
   for(let f=0;f<15;f++){
    const delta=parts[q].feature[f]-parts[p].feature[f],a=model.links[p*2+dir];
    a[f]+=(delta-a[f])/n;
   }
  }
 }
 model.count=n;return model;
}
function distance(a,b){let v=0;for(let k=0;k<15;k++){const d=a[k]-b[k];v+=d*d*(k<3?1.7:k<12?.65:.5);}return v/15;}
function join(a,b,dir){
 // One-pixel boundary continuity; this is a NONLEARNED control.
 let s=0;
 for(let i=0;i<TILE;i++)for(let c=0;c<3;c++){
  const ai=dir===0?(i*TILE+TILE-1)*4+c:((TILE-1)*TILE+i)*4+c;
  const bi=dir===0?(i*TILE)*4+c:i*4+c;
  const d=(a.pix[ai]-b.pix[bi])/255;s+=d*d;
 }
 return s/(3*TILE);
}
function shuffle(n,rnd){const a=Array.from({length:n},(_,i)=>i);for(let i=n-1;i>0;i--){const j=Math.floor(rnd()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function hungarian(cost){ // Assignment uses learned slot prototypes, never source tile indices.
 const n=cost.length,u=Array(n+1).fill(0),v=Array(n+1).fill(0),p=Array(n+1).fill(0),way=Array(n+1).fill(0);
 for(let i=1;i<=n;i++){
  p[0]=i;let j0=0;const minv=Array(n+1).fill(Infinity),used=Array(n+1).fill(false);
  do{
   used[j0]=true;const i0=p[j0];let delta=Infinity,j1=0;
   for(let j=1;j<=n;j++)if(!used[j]){
    const cur=cost[i0-1][j-1]-u[i0]-v[j];
    if(cur<minv[j]){minv[j]=cur;way[j]=j0;}
    if(minv[j]<delta){delta=minv[j];j1=j;}
   }
   for(let j=0;j<=n;j++)if(used[j]){u[p[j]]+=delta;v[j]-=delta;}
   else minv[j]-=delta;
   j0=j1;
  }while(p[j0]!==0);
  do{const j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0!==0);
 }
 const ans=Array(n);for(let j=1;j<=n;j++)ans[j-1]=p[j]-1;return ans;
}
export function energy(board,parts,model,{memory=1,relation=.6,seam=.45}={}){
 const confidence=Math.min(1,model.count/8);let pos=0,rel=0,edge=0;
 for(let p=0;p<COUNT;p++){
  const a=parts[board[p]];
  if(model.count>0)pos+=distance(a.feature,model.slots[p]);
  for(let dir=0;dir<2;dir++){
   const q=dir===0?(p%GRID===GRID-1?-1:p+1):(p>=COUNT-GRID?-1:p+GRID);
   if(q<0)continue;
   const b=parts[board[q]];
   edge+=join(a,b,dir);
   if(model.count>0){
    const expected=model.links[p*2+dir];let d=0;
    for(let k=0;k<15;k++)d+=sq((b.feature[k]-a.feature[k])-expected[k]);
    rel+=d/15;
   }
  }
 }
 return {position:pos/COUNT,relation:rel/(COUNT*2-2*GRID),seam:edge/(COUNT*2-2*GRID),
  total:confidence*(memory*pos/COUNT+relation*rel/(COUNT*2-2*GRID))+seam*edge/(COUNT*2-2*GRID)};
}
export function accuracy(board,parts){let ok=0;for(let p=0;p<COUNT;p++)if(parts[board[p]].truth===p)ok++;return ok/COUNT;}
export function solve(model,parts,opts={},seed=1){
 const w={memory:opts.memory??1,relation:opts.relation??.6,seam:opts.seam??.45};
 const rand=random(seed);let best=null,bestE=Infinity;
 const runs=model.count>0&&w.memory>0?2:4;
 for(let run=0;run<runs;run++){
  let board=shuffle(COUNT,rand);
  if(run===0&&model.count>0&&w.memory>0){
   const cost=parts.map(t=>model.slots.map(m=>distance(t.feature,m)));
   // Each row = tile; each column = destination slot.
   const tileToSlot=hungarian(cost),placed=Array(COUNT);
   for(let ti=0;ti<COUNT;ti++)placed[tileToSlot[ti]]=ti;
   board=placed;
  }
  let e=energy(board,parts,model,w).total;
  if(e<bestE){bestE=e;best=board.slice();}
  const iterations=model.count>0&&w.memory>0?1600:3300;
  for(let step=0;step<iterations;step++){
   const i=Math.floor(rand()*COUNT),j=Math.floor(rand()*COUNT);if(i===j)continue;
   [board[i],board[j]]=[board[j],board[i]];
   const trial=energy(board,parts,model,w).total;
   const T=.004*Math.pow(.008,step/iterations)+.00001;
   if(trial<=e||rand()<Math.exp((e-trial)/T)){e=trial;}
   else [board[i],board[j]]=[board[j],board[i]];
   if(e<bestE){bestE=e;best=board.slice();}
  }
 }
 return {board:best,energy:energy(best,parts,model,w),accuracy:accuracy(best,parts),weights:w};
}
export function scrambled(parts,seed=10){return shuffle(parts.length,random(seed));}
export function advance(board,target){const i=board.findIndex((t,p)=>t!==target[p]);if(i<0)return false;
 const j=board.indexOf(target[i]);[board[i],board[j]]=[board[j],board[i]];return true;}
export function drawParts(parts,board){
 const d=new Uint8ClampedArray(SIDE*SIDE*4);
 for(let p=0;p<COUNT;p++){
  const patch=parts[board[p]].pix,x=(p%GRID)*TILE,y=Math.floor(p/GRID)*TILE;
  for(let iy=0;iy<TILE;iy++)d.set(patch.subarray(iy*TILE*4,(iy+1)*TILE*4),((y+iy)*SIDE+x)*4);
 }
 return {width:SIDE,height:SIDE,data:d};
}
export function prototypeImage(model){
 const d=new Uint8ClampedArray(SIDE*SIDE*4);
 for(let p=0;p<COUNT;p++){
  const f=model.slots[p],x=(p%GRID)*TILE,y=Math.floor(p/GRID)*TILE;
  for(let iy=0;iy<TILE;iy++)for(let ix=0;ix<TILE;ix++){
   const z=((y+iy)*SIDE+x+ix)*4; const l=f[3+Math.floor(iy*3/TILE)*3+Math.floor(ix*3/TILE)];
   const mean=.299*f[0]+.587*f[1]+.114*f[2];
   for(let c=0;c<3;c++)d[z+c]=clamp(255*(f[c]+(l-mean)*.8));
   d[z+3]=255;
  }
 }
 return {width:SIDE,height:SIDE,data:d};
}
