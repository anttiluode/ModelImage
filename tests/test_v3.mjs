import assert from 'node:assert/strict';
import {makeWorld} from '../world-core.js';
import {P,G,CELLS,random,patchPixels,sieve,synthesize,ResidueWorld,maskRectangle,maskedImage,maskedMSE,naiveCompletion} from '../v3-core.js';
assert.equal(CELLS,144);
const r=random(1022);
for(let k=0;k<8;k++){
 const x=new Float32Array(P*P*3);
 for(let i=0;i<x.length;i++)x[i]=r();
 const bands=sieve(x),got=synthesize(...bands);
 assert.deepEqual(bands.map(x=>x.length),[12,48,192]);
 let max=0;for(let i=0;i<x.length;i++)max=Math.max(max,Math.abs(got[i]-x[i]));
 assert.ok(max<2e-7,'three-band sieve must be invertible');
}
const m=new ResidueWorld({caps:[32,48,64]});
for(let i=0;i<95;i++)m.teach(makeWorld(101+i*43,i%2?'parity':'opposite'));
let summary=m.summary();console.log('ModelImage V3 atoms and learned relationship counts:',summary);
assert.equal(summary.lessons,95);
assert.ok(summary.atoms.every(x=>x>0&&x<=64));
assert.ok(summary.spatialPairs>10&&summary.spatialTriples>10);
const test=makeWorld(99123,'parity'),mask=maskRectangle(3,4,4,4);
assert.equal(mask.length,16);
const obs=maskedImage(test,mask);
for(const deformation of [0,1]){
 for(const order of [1,2]){
  for(const detail of [0,1]){
   const a=m.complete(obs,mask,{order,detail,deformation});
   const b=m.complete(test,mask,{order,detail,deformation});
   assert.deepEqual(Array.from(a.image.data),Array.from(b.image.data),'do not read hidden pixels');
   assert.equal(a.codes.length,CELLS);
   assert.equal(a.provenance.length,CELLS);
   assert.ok(Number.isFinite(maskedMSE(a.image,test,mask)));
   // Visible pixels must remain exact.
   const hidden=new Set(mask);
   for(let p=0;p<CELLS;p++)if(!hidden.has(p)){
    const x=(p%G)*P,y=Math.floor(p/G)*P;
    for(let iy=0;iy<P;iy++)for(let ix=0;ix<P;ix++){
     const i=((y+iy)*96+x+ix)*4;
     for(let c=0;c<3;c++)assert.equal(a.image.data[i+c],test.data[i+c]);
    }
   }
  }
 }
}
const naive=naiveCompletion(obs,mask);
assert.ok(Number.isFinite(maskedMSE(naive,test,mask)));
const nothing=new ResidueWorld().complete(obs,mask,{deformation:1});
assert.ok(Number.isFinite(maskedMSE(nothing.image,test,mask)));
console.log('ModelImage V3: all hierarchy, competitive learning, deformation, and nonleakage checks passed');
