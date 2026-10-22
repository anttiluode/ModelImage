import assert from 'node:assert/strict';
import {GRID,SIDE,COUNT,random,synthetic,tiles,emptyModel,teach,solve,accuracy,advance,scrambled,drawParts,prototypeImage} from '../core.mjs';
assert.equal(COUNT,36);
for(const theme of ['face','garden','city','abstract']){
 const img=synthetic(theme,42),parts=tiles(img);
 assert.equal(img.data.length,SIDE*SIDE*4);
 assert.equal(parts.length,COUNT);
 const board=Array.from({length:COUNT},(_,i)=>i);
 assert.equal(accuracy(board,parts),1);
 assert.deepEqual(Array.from(drawParts(parts,board).data),Array.from(img.data));
 const a=emptyModel();
 for(let s=1;s<=16;s++)teach(a,synthetic(theme,s*13+7));
 assert.equal(a.count,16);
 assert.equal(prototypeImage(a).data.length,img.data.length);
 const b=solve(a,parts,{memory:1.2,relation:.6,seam:.45},42);
 const c=solve(emptyModel(),parts,{memory:0,relation:0,seam:.45},42);
 let cur=scrambled(parts,8),moves=0;
 while(advance(cur,b.board)){moves++;assert.ok(moves<=COUNT);}
 assert.deepEqual(cur,b.board);
 console.log(theme,{learned:b.accuracy,edgeOnly:c.accuracy,initial:accuracy(scrambled(parts,8),parts),moves});
}
console.log('ModelImage core tests passed');
