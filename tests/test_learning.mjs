import assert from 'node:assert/strict';
import {synthetic,tiles,emptyModel,teach,solve,COUNT,energy,accuracy,drawParts} from '../core.mjs';
function train(theme,n=20){
 const model=emptyModel();
 for(let s=0;s<n;s++)teach(model,synthetic(theme,103+s*997));
 return model;
}
function mse(parts,board,original){const a=drawParts(parts,board).data,b=original.data;let sum=0;
 for(let i=0;i<a.length;i+=4)for(let c=0;c<3;c++)sum+=(a[i+c]-b[i+c])**2;
 return sum/(original.width*original.height*3*255*255);
}
for(const theme of ['face','garden','city','abstract']){
 const model=train(theme),untrained=emptyModel();
 let learnedAcc=0,seamAcc=0,learnedMse=0,seamMse=0;
 for(let k=1;k<=2;k++){
  const img=synthetic(theme,500001+k*911),p=tiles(img);
  // Evaluation ground-truth indices are scrambled here to ensure the solver ignores them.
  const stored=p.map(x=>x.truth);for(const x of p)x.truth=-12345;
  const solution=solve(model,p,{memory:1.2,relation:.6,seam:.45},707+k*13);
  const baseline=solve(untrained,p,{memory:0,relation:0,seam:.45},707+k*13);
  p.forEach((x,i)=>x.truth=stored[i]);
  learnedAcc+=accuracy(solution.board,p);seamAcc+=accuracy(baseline.board,p);
  learnedMse+=mse(p,solution.board,img);seamMse+=mse(p,baseline.board,img);
  assert.ok(Number.isFinite(solution.energy.total));assert.equal(solution.board.length,COUNT);
  assert.equal(new Set(solution.board).size,COUNT);
 }
 console.log(theme+' heldout avg: trained positions '+(100*learnedAcc/2).toFixed(1)+'%, edge-only '+(100*seamAcc/2).toFixed(1)+'%; pixel MSE '+(learnedMse/2).toFixed(5)+' vs '+(seamMse/2).toFixed(5));
 if(['face','city','abstract'].includes(theme))assert.ok(learnedAcc>seamAcc,'training should help on '+theme+' heldout pair');
}
console.log('Learning ablations passed (garden printed without asserting improvement).');
