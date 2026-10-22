import assert from "node:assert/strict";
import {GRID,NSITES,makeWorld,MovingSieve,halfMask,maskedError,makeMasked} from "../world-core.mjs";

function train(world, n=360){
 const m=new MovingSieve();
 for(let i=0;i<n;i++)m.teach(makeWorld(100003+997*i,world));
 return m;
}
let pass=true;
for(const world of ["parity","opposite","copy"]){
 const m=train(world);
 const sample=makeWorld(881777,world);
 const {codes,face}=sample;
 assert.equal(m.lessons,360);
 assert.ok(m.atoms.length>=7&&m.atoms.length<=24);
 let right2=0,right1=0,n=120, mse2=0,mse1=0,notLeaked=0;
 let relationExample;
 for(let k=0;k<n;k++){
  const t=makeWorld(500004+k*913,world),h=[t.face.mouth];
  const p2=m.complete(t,h,2),p1=m.complete(t,h,1);
  const expected=m.encode(t)[h[0]];
  right2+=+(p2.codes[h[0]]===expected);right1+=+(p1.codes[h[0]]===expected);
  mse2+=maskedError(p2.image,t,h);mse1+=maskedError(p1.image,t,h);
  relationExample??=p2.provenance[h[0]];
  // The learner must not inspect a hidden tile at inference time.
  const blank=makeMasked(t,h),q=m.complete(blank,h,2);
  assert.equal(q.codes[h[0]],p2.codes[h[0]]);
  notLeaked++;
 }
 console.log(JSON.stringify({world,atoms:m.atoms.length,lessons:m.lessons,triplet:right2/n,single:right1/n,tripletMSE:mse2/n,singleMSE:mse1/n,example:relationExample}));
 if(world==="parity"||world==="opposite"){
  if(right2/n<.85)pass=false;
  if(right2<=right1+20)pass=false;
 }
 if(world==="copy"&&right1/n<.8)pass=false;
 assert.equal(notLeaked,n);
 const large=makeWorld(707222,world);
 for(const mask of [halfMask(3),halfMask(4)]){
  assert.equal(mask.length,18);
  const out=m.complete(large,mask,2);
  assert.equal(out.image.data.length,large.data.length);
  assert.ok(Number.isFinite(maskedError(out.image,large,mask)));
  for(const p of mask)assert.ok(out.codes[p]>=0);
 }
}
assert.ok(pass,"Higher-order predictive relation failed to outperform pairwise on heldout parity.");
console.log("Moving-part discovery tests passed.");
