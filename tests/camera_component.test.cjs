/**
 * Browser-recorder lifecycle checks using mocked devices, without accessing a real
 * webcam, microphone, or browser. The production component script runs unchanged in
 * a VM with a minimal DOM/media boundary. Tests verify explicit permission requests,
 * recorded-clip submission, track cleanup, and recoverable denied-permission state.
 * Node is only a development test runner; the Python app needs no npm/build step.
 */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../src/autotab/ui/camera_component/index.html'),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];

/** Create isolated DOM/device fakes and run the exact bundled recorder script. */
function harness(deny=false) {
  const elements={};
  for(const id of ['preview','devices','enable','start','stop','release','status']) {
    elements[id]={disabled:['devices','start','stop','release'].includes(id),value:'',
      textContent:'',replaceChildren(){},async play(){}};
  }
  let requests=0,stopped=0;
  const constraints=[],messages=[];
  const tracks=[{stop(){stopped++;},getSettings(){return {deviceId:'camera1'};}},
                {stop(){stopped++;}}];
  const stream={getTracks:()=>tracks,getVideoTracks:()=>[tracks[0]]};
  class FakeRecorder {
    static isTypeSupported(type){return type.startsWith('video/webm');}
    constructor(input,options){assert.equal(input,stream);this.mimeType=options.mimeType;this.state='inactive';}
    start(){this.state='recording';}
    stop(){this.state='inactive';this.ondataavailable({data:new Blob(['video'],{type:this.mimeType})});this.completion=this.onstop();}
  }
  class FakeReader {
    readAsDataURL(){this.result='data:video/webm;base64,dmlkZW8=';this.onload();}
  }
  const parent={postMessage(message){messages.push(message);}};
  const context={document:{getElementById:id=>elements[id],body:{scrollHeight:400}},parent,
    window:{MediaRecorder:FakeRecorder,addEventListener(){}},MediaRecorder:FakeRecorder,
    navigator:{mediaDevices:{async getUserMedia(options){
      requests++;constraints.push(options);
      if(deny){const error=new Error('denied');error.name='NotAllowedError';throw error;}
      return stream;
    },async enumerateDevices(){return [{kind:'videoinput',deviceId:'camera1',label:'Test camera'}];}}},
    Blob,FileReader:FakeReader,Option:class {},ResizeObserver:class {observe(){}},
    crypto:{randomUUID:()=> 'capture-id'},setInterval(){return 1;},clearInterval(){},Date};
  vm.createContext(context);vm.runInContext(script,context);
  return {elements,context,messages,constraints,get requests(){return requests;},get stopped(){return stopped;}};
}

test('does not request hardware until Enable is clicked', async()=>{
  const h=harness();assert.equal(h.requests,0);
  await h.elements.enable.onclick();
  assert.equal(h.requests,1);
  assert.equal(h.constraints[0].audio.echoCancellation,false);
  assert.ok(h.constraints[0].video);
  assert.equal(h.elements.start.disabled,false);
});

test('Stop submits a completed clip and releases both tracks', async()=>{
  const h=harness();await h.elements.enable.onclick();h.elements.start.onclick();
  assert.equal(h.elements.stop.disabled,false);
  h.elements.stop.onclick();await vm.runInContext('recorder.completion',h.context);
  assert.equal(h.stopped,2);
  const message=h.messages.find(m=>m.type==='streamlit:setComponentValue');
  assert.equal(message.value.data,'dmlkZW8=');
  assert.equal(message.value.id,'capture-id');
  assert.equal(h.elements.enable.disabled,false);
  assert.equal(h.elements.preview.srcObject,null);
});

test('permission denial is visible and can be retried', async()=>{
  const h=harness(true);await h.elements.enable.onclick();
  assert.match(h.elements.status.textContent,/permission was denied/);
  assert.equal(h.elements.enable.disabled,false);
  assert.equal(h.elements.start.disabled,true);
});

test('Turn camera off releases devices without submitting a recording', async()=>{
  const h=harness();await h.elements.enable.onclick();h.elements.release.onclick();
  assert.equal(h.stopped,2);
  assert.equal(h.messages.some(m=>m.type==='streamlit:setComponentValue'),false);
});
