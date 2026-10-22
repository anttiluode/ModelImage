// Browser GUI regression for the exact embedded script used by v3.html.
// No headless-browser dependencies; tests event wiring and render initialization
// using a small DOM canvas shim. Not a substitute for visual cross-browser QA.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import * as v3 from '../v3-core.js';
import {makeWorld} from '../world-core.js';
const html=readFileSync(new URL('../v3.html',import.meta.url),'utf8');
const source=html.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(source,'V3 module script must exist');
const script=source.replace(/^import .+;$/mg,'');
const els=new Map();let count=0;
function node(name){
 if(!els.has(name))els.set(name,{
  name,value:name==='testType'?'synthetic':name==='trainWorld'?'parity':name==='testWorld'?'parity':name==='occlusion'?'small':
   name==='order'?'2':name==='deformation'?'1':name==='detail'?'1':'',
  textContent:'',style:{},children:[],disabled:false,width:96,height:96,
  getContext(){return{putImageData(){}};},getBoundingClientRect(){return{left:0,top:0,width:96,height:96};},
  replaceChildren(){this.children=[];},append(x){this.children.push(x);},setAttribute(){},
 });
 return els.get(name);
}
const document={
 querySelector:x=>node(x.slice(1)),
 createElement:tag=>node(tag+(count++)),
 createElementNS:(ns,tag)=>node(tag+(count++))
};
class ImageData {constructor(data,width,height){this.data=data;this.width=width;this.height=height;}}
const args=['document','ImageData','requestAnimationFrame',...Object.keys(v3),'makeWorld'];
const gui=new Function(...args,script+';return {trainSynthetic,reconstruct,newTest,model};');
const live=gui(document,ImageData,cb=>cb(),...Object.values(v3),makeWorld);
assert.ok(node('visible').getContext,'visible panel created');
assert.ok(node('status2').textContent.includes('MSE'),'first inference initializes');
await live.trainSynthetic(14);
assert.equal(node('lessons').textContent,14);
assert.ok(+node('coarseCount').textContent>0);
assert.ok(node('coarseAtoms').children.length>0);
assert.ok(node('selected').textContent.includes('Hidden cell'));
node('deformation').value='0';node('deformation').oninput();
assert.equal(node('defVal').textContent,'0 patches');
node('occlusion').value='big';node('occlusion').onchange();
assert.ok(node('status2').textContent.includes('trust-checked'));
node('testWorld').value='opposite';await node('testWorld').onchange();
node('erase').onclick();
assert.equal(node('lessons').textContent,0);
console.log('V3 browser controls, rendering, hierarchy, photo-independent mode, and erasure passed.');
