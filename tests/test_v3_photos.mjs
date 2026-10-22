import assert from 'node:assert/strict';
import {readdirSync,readFileSync} from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {ResidueWorld,maskRectangle,maskedImage,maskedMSE,naiveCompletion,completeWithTrust} from '../v3-core.js';
const root=process.argv[2]||path.join(os.tmpdir(),'modelimage-v3-photos');
function readPPM(file){
 const raw=readFileSync(file);let at=0;
 function word(){while(raw[at]===10||raw[at]===13||raw[at]===32)at++;let s='';
  while(at<raw.length&&raw[at]!==10&&raw[at]!==13&&raw[at]!==32)s+=String.fromCharCode(raw[at++]);
  return s;
 }
 assert.equal(word(),'P6');const w=+word(),h=+word(),max=+word();
 assert.equal(w,96);assert.equal(h,96);assert.equal(max,255);
 at++; // consume exactly one PPM whitespace delimiter; do not skip possible pixel values 10/13.
 const d=new Uint8ClampedArray(w*h*4);
 for(let i=0;i<w*h;i++)for(let c=0;c<3;c++)d[i*4+c]=raw[at+i*3+c];
 for(let i=0;i<w*h;i++)d[i*4+3]=255;
 return {width:w,height:h,data:d};
}
function load(split){return readdirSync(path.join(root,split)).filter(s=>s.endsWith('.ppm')).sort().map(name=>({name,image:readPPM(path.join(root,split,name))}));}
const train=load('train'),test=load('test');
assert.equal(train.length,96);assert.equal(test.length,12);
const names=new Set(train.map(x=>x.name.split('_')[0]));
assert.ok(test.every(x=>!names.has(x.name.split('_')[0])),'PHOTO ID leakage');
const model=new ResidueWorld({caps:[48,64,80],seed:17});
for(const item of train)model.teach(item.image);
const summary=model.summary();
let totals={two:0,one:0,coarse:0,naive:0,empty:0,trusted:0};
let trustSum=0,trustNonzero=0;
const untrained=new ResidueWorld(),byPhoto={};let n=0;
for(const item of test){
 const id=item.name.split('_')[0],testMasks=[maskRectangle(4,4,4,4),maskRectangle(2,6,4,4)];
 byPhoto[id]??={two:0,one:0,coarse:0,naive:0,n:0};
 for(const h of testMasks){
  const visible=maskedImage(item.image,h);
  const two=model.complete(visible,h,{order:2,detail:1});
  const one=model.complete(visible,h,{order:1,detail:1});
  const coarse=model.complete(visible,h,{order:2,detail:0});
  const naive=naiveCompletion(visible,h);
  const empty=untrained.complete(visible,h,{order:2,detail:1});
  const trusted=completeWithTrust(model,visible,h,{order:2,detail:1,deformation:1});
  trustSum+=trusted.trust;trustNonzero+=Number(trusted.trust>0);
  const evaluation={two:two.image,one:one.image,coarse:coarse.image,naive,empty:empty.image,trusted:trusted.image};
  for(const [key,prediction] of Object.entries(evaluation)){
   const err=maskedMSE(prediction,item.image,h);
   totals[key]+=err;
   if(byPhoto[id][key]!==undefined)byPhoto[id][key]+=err;
  }
  // Nonleakage: completing an intact image with the SAME hidden mask must not change the output.
  const alternateTrust=completeWithTrust(model,item.image,h,{order:2,detail:1,deformation:1});
  assert.deepEqual([...alternateTrust.image.data],[...trusted.image.data],'Trust must not read the unknown region');
  const same=model.complete(item.image,h,{order:2,detail:1});
  assert.deepEqual([...two.image.data],[...same.image.data],'Hidden region was accessed by model');
  const covered=new Set(h);
  for(let p=0;p<144;p++)if(!covered.has(p)){
   const x=p%12*8,y=Math.floor(p/12)*8;
   for(let iy=0;iy<8;iy++)for(let ix=0;ix<8;ix++){
    const k=((y+iy)*96+x+ix)*4;
    for(let c=0;c<3;c++)assert.equal(two.image.data[k+c],item.image.data[k+c],'Visible pixel changed');
   }
  }
  byPhoto[id].n++;n++;
 }
}
for(const k in totals)totals[k]/=n;
for(const k in byPhoto)for(const method of ['two','one','coarse','naive'])byPhoto[k][method]/=byPhoto[k].n;
console.log(JSON.stringify({benchmark:'disjoint-photographs',trainImages:train.length,testImages:test.length,maskedTests:n,learned:summary,meanMSE:totals,trust:{mean:trustSum/n,nonzero:trustNonzero},perPhoto:byPhoto},null,2));
assert.ok(Object.values(totals).every(x=>Number.isFinite(x)));
console.log('Real-photo transfer test complete (no accuracy improvement presumed).');
