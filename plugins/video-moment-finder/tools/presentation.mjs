/** Editable PPTX draft exporter. Requires the host's @oai/artifact-tool runtime. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const [input, output, font='Arial'] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: node presentation.mjs prepared.json build-directory [font]');
const data=JSON.parse(await fs.readFile(input,'utf8'));
if(data.format!=='vmf-presentation-1'||!Array.isArray(data.slides)||data.slides.length<3||data.slides.length>16) throw new Error('Use tools/presentation.py to prepare 3–16 slides');
const out=path.resolve(output);await fs.mkdir(out,{recursive:true});
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const authoredSlides=[];
const colors={ink:'#17283F',blue:'#245BDD',teal:'#087E83',muted:'#52637A'};
function text(slide,value,x,y,w,h,size,color=colors.ink,bold=false){
  if(typeof value!=='string')throw new Error('Expected text');
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;shape.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none',wrap:true,insets:{left:0,right:0,top:0,bottom:0}};return shape;
}
for(const [i,s] of data.slides.entries()){
  if(!['cover','statement','evidence','comparison','exercise'].includes(s.layout)||!Array.isArray(s.body)||s.body.length>3||!Array.isArray(s.evidence))throw new Error('Invalid prepared slide');
  const slide=deck.slides.add(),cover=s.layout==='cover',exercise=s.layout==='exercise',ink=cover?'#FFFFFF':colors.ink,muted=cover?'#BFCEE4':colors.muted,accent=cover?'#9DBBFF':exercise?colors.teal:colors.blue;
  authoredSlides.push(slide);
  slide.background.fill=cover?colors.ink:exercise?'#E7F2F2':'#FFFFFF';
  text(slide,s.label.toUpperCase(),72,48,1120,30,17,muted,true);
  text(slide,s.title,72,102,1130,cover?175:124,cover?66:48,ink,true);
  let y=cover?300:256;
  if(s.layout==='evidence' && s.image){
    if(!/^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=]+$/.test(s.image))throw new Error('Only prepared embedded PNG/JPEG frames are supported');
    if(s.headline){text(slide,s.headline,72,y,470,88,52,accent,true);y+=102}
    for(const p of s.body){text(slide,p,72,y,460,120,27,ink);y+=132}
    slide.images.add({dataUrl:s.image,alt:s.image_alt,fit:'contain',position:{left:582,top:258,width:626,height:352}});
    text(slide,s.image_caption,582,618,626,43,15,muted);
  }else if(s.layout==='comparison'){
    text(slide,s.body.join('\n'),72,240,1120,78,26,muted);
    if(!Array.isArray(s.columns)||s.columns.length!==2)throw new Error('Comparison needs two columns');
    for(const [j,c] of s.columns.entries()){const x=72+j*585;text(slide,c.title,x,350,525,75,38,j?colors.teal:colors.blue,true);text(slide,c.body,x,442,520,170,29,ink)}
  }else{
    if(s.headline){text(slide,s.headline,72,y,1120,118,Math.min(cover?68:72,Math.max(40,Math.floor(2100/s.headline.length))),accent,true);y+=142}
    for(const p of s.body){text(slide,p,72,y,cover?1030:1120,92,cover?30:29,ink);y+=108}
    if(s.image_missing)text(slide,s.image_missing,72,602,1080,44,17,muted);
  }
  text(slide,String(i+1).padStart(2,'0'),1160,670,48,25,16,muted);
  const refs=s.evidence.map(e=>e.id+' '+e.time).join(' · ');
  text(slide,refs,72,672,1060,25,14,muted);
  slide.speakerNotes.textFrame.setText(s.speaker_notes);
}
await (await PresentationFile.exportPptx(deck)).save(path.join(out,'candidate.pptx'));
for(const [i,slide] of authoredSlides.entries()){
  const png=await deck.export({slide,format:'png',scale:1});await fs.writeFile(path.join(out,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
  const layout=await slide.export({format:'layout'});await fs.writeFile(path.join(out,`slide-${i+1}.layout.json`),await layout.text());
}
process.stdout.write(JSON.stringify({draft:path.join(out,'candidate.pptx'),slides:data.slides.length,finalized:false})+'\n');
