'use strict';
const $ = id => document.getElementById(id);
let jobId=null, tracks={}, frame=0, playing=false, started=0, startFrame=0;
let eventStream=null, localVideoUrl=null, elapsedTimer=null, elapsedOrigin=0, modelReady=true, inputBusy=false;
let loadRequests={};
let poseRadius=1;
let theme='dark';
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(40,1,.01,100);
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setClearColor(0x0b1220);renderer.setPixelRatio(Math.min(devicePixelRatio,2));
$('view3d').appendChild(renderer.domElement);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;
const group=new THREE.Group();scene.add(group);
const grid=new THREE.GridHelper(4,20,0x34465b,0x213042);scene.add(grid);
const bones=[], joints=[];
for(let i=0;i<16;i++){const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]),new THREE.LineBasicMaterial({color:0x3b82f6}));bones.push(line);group.add(line);}
for(let i=0;i<17;i++){const mesh=new THREE.Mesh(new THREE.SphereGeometry(.022,10,8),new THREE.MeshBasicMaterial({color:0x93c5fd}));joints.push(mesh);group.add(mesh);}
group.visible=false;
function fitDistance(aspect){const halfFov=THREE.MathUtils.degToRad(camera.fov/2),limit=Math.min(halfFov,Math.atan(Math.tan(halfFov)*aspect));return poseRadius*1.15/Math.sin(limit);}
function resetCamera(){const distance=tracks['3d']?fitDistance(camera.aspect):Math.sqrt(14);camera.position.copy(new THREE.Vector3(2,1,3).normalize().multiplyScalar(distance));controls.target.set(0,0,0);controls.update();}
resetCamera();$('reset').onclick=resetCamera;
const context=$('canvas2d').getContext('2d');
function resize(){const box=$('view3d').getBoundingClientRect();if(!box.width||!box.height)return;renderer.setSize(box.width,box.height);const aspect=box.width/box.height;if(tracks['3d'])camera.position.sub(controls.target).multiplyScalar(fitDistance(aspect)/fitDistance(camera.aspect)).add(controls.target);camera.aspect=aspect;camera.updateProjectionMatrix();const b=$('canvas2d').getBoundingClientRect();$('canvas2d').width=b.width*devicePixelRatio;$('canvas2d').height=b.height*devicePixelRatio;draw();}
new ResizeObserver(resize).observe($('view3d'));new ResizeObserver(resize).observe($('canvas2d').parentElement);
function fps(){return tracks['2d']?.fps||tracks['3d']?.fps||30;}
function count(){return Math.max(tracks['2d']?.frames.length||0,tracks['3d']?.frames.length||0);}
function draw(){
 const canvas=$('canvas2d'),w=canvas.width,h=canvas.height;context.clearRect(0,0,w,h);
 const t=tracks['2d'];if(t){const points=t.frames[Math.min(frame,t.frames.length-1)],scale=Math.min(w/t.width,h/t.height)*.9,ox=(w-t.width*scale)/2,oy=(h-t.height*scale)/2;context.lineWidth=2.5*devicePixelRatio;context.strokeStyle=theme==='light'?'#007d96':'#22d3ee';
 for(const [a,b] of t.bones){if(points[a][2]<.1||points[b][2]<.1)continue;context.beginPath();context.moveTo(points[a][0]*scale+ox,points[a][1]*scale+oy);context.lineTo(points[b][0]*scale+ox,points[b][1]*scale+oy);context.stroke();}
 for(const p of points){if(p[2]<.1)continue;context.fillStyle=theme==='light'?'#245fae':'#f8fafc';context.beginPath();context.arc(p[0]*scale+ox,p[1]*scale+oy,3*devicePixelRatio,0,Math.PI*2);context.fill();}}
 const t3=tracks['3d'];if(t3){const raw=t3.frames[Math.min(frame,t3.frames.length-1)],root=raw[0];const points=raw.map(p=>new THREE.Vector3(p[0]-root[0],-(p[1]-root[1]),-(p[2]-root[2])));points.forEach((p,i)=>joints[i].position.copy(p));t3.bones.forEach(([a,b],i)=>{bones[i].geometry.setFromPoints([points[a],points[b]]);});group.visible=true;}
 $('timeline').value=frame;$('frame').value=`${frame} / ${Math.max(0,count()-1)} · ${fps().toFixed(2)} FPS`;
}
function applyTheme(value){
 theme=value==='light'?'light':'dark';
 document.documentElement.dataset.theme=theme;
 $('toggleTheme').textContent=theme==='light'?'暗色模式':'亮色模式';
 $('toggleTheme').setAttribute('aria-pressed',String(theme==='light'));
 renderer.setClearColor(theme==='light'?0xedf2f8:0x0b1220);
 bones.forEach(line=>line.material.color.setHex(theme==='light'?0x245fae:0x3b82f6));
 joints.forEach(joint=>joint.material.color.setHex(theme==='light'?0x286acc:0x93c5fd));
 const color=new THREE.Color(theme==='light'?0xb9c8da:0x2a384f);
 const colors=grid.geometry.attributes.color;for(let i=0;i<colors.count;i++)colors.setXYZ(i,color.r,color.g,color.b);colors.needsUpdate=true;
 try{localStorage.setItem('kps-extractor-theme',theme);}catch{}
 draw();
}
let savedTheme='dark';try{savedTheme=localStorage.getItem('kps-extractor-theme')||savedTheme;}catch{}
applyTheme(savedTheme);
$('toggleTheme').onclick=()=>applyTheme(theme==='light'?'dark':'light');
function showInspector(visible){$('workspace').classList.toggle('inspector-collapsed',!visible);$('toggleInspector').setAttribute('aria-expanded',String(visible));}
showInspector(!matchMedia('(max-width:900px)').matches);
$('toggleInspector').onclick=()=>showInspector($('workspace').classList.contains('inspector-collapsed'));
$('closeInspector').onclick=()=>showInspector(false);
const tabs=[...document.querySelectorAll('[role="tab"]')];
function selectTab(tab){for(const item of tabs){const selected=item===tab;item.setAttribute('aria-selected',String(selected));item.tabIndex=selected?0:-1;$(item.dataset.panel).hidden=!selected;}}
tabs.forEach((tab,index)=>{tab.onclick=()=>selectTab(tab);tab.onkeydown=e=>{let next;if(e.key==='ArrowRight')next=(index+1)%tabs.length;else if(e.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;else if(e.key==='Home')next=0;else if(e.key==='End')next=tabs.length-1;else return;e.preventDefault();selectTab(tabs[next]);tabs[next].focus();};});
function stop(){playing=false;$('play').textContent='播放';$('video').pause();}
function seek(value){stop();frame=Math.max(0,Math.min(count()-1,value));if($('video').readyState>=1)$('video').currentTime=frame/fps();draw();}
function start(){if(!count())return;if(frame>=count()-1)frame=0;playing=true;started=performance.now();startFrame=frame;$('play').textContent='暂停';const video=$('video');if(video.readyState>=1){video.currentTime=frame/fps();video.playbackRate=Number($('speed').value);video.play().catch(()=>{});}}
$('play').onclick=()=>playing?stop():start();$('prev').onclick=()=>seek(frame-1);$('next').onclick=()=>seek(frame+1);$('timeline').oninput=e=>seek(Number(e.target.value));$('speed').onchange=()=>{if(playing){stop();start();}};
$('video').addEventListener('seeking',()=>{if(!playing&&count()){frame=Math.min(count()-1,Math.round($('video').currentTime*fps()));draw();}});
$('video').addEventListener('play',()=>{if(!playing&&count()){frame=Math.min(count()-1,Math.floor($('video').currentTime*fps()));playing=true;started=performance.now();startFrame=frame;$('play').textContent='暂停';}});
$('video').addEventListener('pause',()=>{if(playing){playing=false;$('play').textContent='播放';}});
$('video').addEventListener('error',()=>{$('videoEmpty').hidden=false;$('videoEmpty').textContent='浏览器不支持此视频编码，骨架仍可查看';});
document.addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName)||!count())return;if(e.code==='Space'){e.preventDefault();playing?stop():start();}if(e.code==='ArrowLeft')seek(frame-1);if(e.code==='ArrowRight')seek(frame+1);});
function animate(now){requestAnimationFrame(animate);if(playing){const video=$('video');const cursor=video.readyState>=2&&!video.paused&&!video.ended?video.currentTime*fps():startFrame+(now-started)/1000*fps()*Number($('speed').value);frame=Math.min(count()-1,Math.floor(cursor));draw();if(frame>=count()-1)stop();}controls.update();renderer.render(scene,camera);}requestAnimationFrame(animate);
async function json(url,options){
 const response=await fetch(url,options),data=await response.json();
 if(!response.ok){const error=Error(data.detail||'请求失败');error.status=response.status;throw error;}
 return data;
}
function formatBytes(value){return value>=1024*1024?`${(value/1024/1024).toFixed(1)} MB`:`${(value/1024).toFixed(1)} KB`;}
function updateInputs(busy){
 inputBusy=busy;
 const file=$('file').files[0];
 $('file').disabled=busy;$('device').disabled=busy;
 $('generate').disabled=busy||!modelReady||!file||file.size>512*1024*1024;
 $('videoDrop').classList.toggle('processing',busy);
 $('chooseAction').textContent=busy?'处理中…':file?'更换视频':'选择视频';
}
function updateElapsed(){const seconds=Math.max(0,Math.floor((performance.now()-elapsedOrigin)/1000));$('elapsed').textContent=`已用时 ${seconds<60?seconds+' 秒':Math.floor(seconds/60)+' 分 '+seconds%60+' 秒'}`;}
function startClock(elapsed=0){clearInterval(elapsedTimer);elapsedOrigin=performance.now()-elapsed*1000;$('elapsed').hidden=false;updateElapsed();elapsedTimer=setInterval(updateElapsed,1000);}
function closeProgress(){if(eventStream){eventStream.close();eventStream=null;}clearInterval(elapsedTimer);elapsedTimer=null;}
function resetResults(){
 closeProgress();jobId=null;stop();tracks={};loadRequests={};frame=0;group.visible=false;
 for(const key of ['play','prev','next','timeline'])$(key).disabled=true;
 for(const key of ['empty2d','empty3d'])$(key).hidden=false;
 $('timeline').max=0;$('downloads').textContent='生成的 NPZ 将显示在这里';$('logs').textContent='尚无日志';
 $('busy').hidden=true;$('progressCount').hidden=true;$('elapsed').hidden=true;draw();
}
$('file').onchange=()=>{
 resetResults();const file=$('file').files[0],selected=!!file;
 if(localVideoUrl){URL.revokeObjectURL(localVideoUrl);localVideoUrl=null;}
 $('videoDrop').classList.toggle('selected',selected);
 $('videoDrop').classList.toggle('invalid',selected&&file.size>512*1024*1024);
 $('selectionIcon').textContent=selected?'✓':'↑';$('selectionTitle').textContent=selected?'已选择视频':'选择运动视频';
 $('selectionHelp').textContent=selected?'可先预览，确认后开始生成':'上传后生成 2D / 3D 骨架';
 $('filename').textContent=file?.name||'';$('filename').hidden=!selected;
 $('fileMeta').textContent=selected?formatBytes(file.size):'';$('fileMeta').hidden=!selected;
 $('previewTitle').textContent=file?.name||'骨架提取预览';
 $('file').setAttribute('aria-label',selected?'更换运动视频':'选择运动视频');
 $('videoEmpty').textContent='选择视频即可预览，无需先上传或生成骨架';$('videoEmpty').hidden=!selected;
 $('video').removeAttribute('src');
 if(selected){localVideoUrl=URL.createObjectURL(file);$('video').src=localVideoUrl;}
 $('video').load();
 $('state').textContent=selected?'视频已选择':'准备就绪';$('state').parentElement.dataset.status='idle';
 $('detail').textContent=selected?'原视频已可预览，点击一键生成骨架开始提取':'选择视频即可预览，确认后开始提取';
 if(selected&&file.size>512*1024*1024){$('state').textContent='视频过大';$('detail').textContent='视频超过 512 MB，请选择较小的视频';}
 if(!modelReady){$('state').textContent='缺少模型';$('detail').textContent='请先放置 MHFormer model_4294.pth（详见 README）';}
 updateInputs(false);
};
$('video').addEventListener('loadedmetadata',()=>{
 if(!localVideoUrl)return;
 $('videoEmpty').hidden=true;const file=$('file').files[0],video=$('video');
 if(file){const duration=Number.isFinite(video.duration)?` · ${video.duration<60?video.duration.toFixed(1)+' 秒':Math.floor(video.duration/60)+':'+String(Math.floor(video.duration%60)).padStart(2,'0')}`:'';
 $('fileMeta').textContent=`${formatBytes(file.size)} · ${video.videoWidth} × ${video.videoHeight}${duration}`;}
});
async function loadTracks(id,files){
 const hadTracks=count()>0;
 for(const slot of ['2d','3d']){
  const marker=slot==='2d'?'2d_h36m':'3d';
  if(tracks[slot]||loadRequests[slot]===id||!files.some(n=>n.includes(marker)))continue;
  loadRequests[slot]=id;
  try{const t=await json(`/api/jobs/${id}/tracks/${slot}`);if(id!==jobId)return;tracks[slot]=t;$(`empty${slot}`).hidden=true;if(slot==='3d'){poseRadius=.2;for(const points of t.frames){const root=points[0];for(const p of points)poseRadius=Math.max(poseRadius,Math.hypot(p[0]-root[0],p[1]-root[1],p[2]-root[2]));}resetCamera();}}
  catch(e){if(id===jobId)$('detail').textContent=`${slot.toUpperCase()} 预览加载失败：${e.message}`;}
  finally{if(loadRequests[slot]===id)delete loadRequests[slot];}
 }
 if(id!==jobId)return;const enabled=count()>0;
 for(const key of ['play','prev','next','timeline'])$(key).disabled=!enabled;
 if(enabled&&!hadTracks&&$('video').readyState>=1){const video=$('video');frame=Math.min(count()-1,Math.floor(video.currentTime*fps()));if(!video.paused&&!video.ended){playing=true;started=performance.now();startFrame=frame;$('play').textContent='暂停';}}
 $('timeline').max=Math.max(0,count()-1);draw();
}
function downloads(id,files,finished){
 const container=$('downloads');container.replaceChildren();
 if(!files.length){container.textContent=finished?'未生成可下载的结果':'等待提取结果';return;}
 for(const name of files){const a=document.createElement('a');a.textContent=name;a.href=`/api/jobs/${id}/files/${encodeURIComponent(name)}`;a.download=name;container.appendChild(a);}
 if(finished){const a=document.createElement('a');a.textContent='下载全部 ZIP';a.href=`/api/jobs/${id}/archive`;container.appendChild(a);}
}
function renderProgress(data){
 const finished=['done','error'].includes(data.status),p=data.progress;
 const stages={initializing:'加载依赖并检查视频',init_2d:'加载 RTMPose / RTMDet（首次下载权重）','2d':'RTMPose 逐帧检测 2D',save_2d:'转换并保存 2D 骨架',init_3d:'加载 MHFormer 权重','3d':'MHFormer 重建 3D',save_3d:'保存 3D 骨架',done:'2D / 3D 均已生成'};
 $('detail').textContent=data.error||p?.message||stages[data.stage]||data.stage;
 $('busy').hidden=finished&&data.status!=='done';$('progressCount').hidden=true;
 if(data.status==='done'){$('busy').value=100;$('progressCount').hidden=false;$('progressCount').textContent='完成';}
 else if(p&&Number.isInteger(p.current)&&p.total>0&&p.current<=p.total){const percent=p.current/p.total*100;$('busy').value=percent;$('progressCount').hidden=false;$('progressCount').textContent=`${p.current} / ${p.total} ${p.unit} · ${percent.toFixed(1)}%`;}
 else{$('busy').removeAttribute('value');if(p?.current>0){$('progressCount').hidden=false;$('progressCount').textContent=`已处理 ${p.current} ${p.unit}`;}}
 elapsedOrigin=performance.now()-(data.elapsed||0)*1000;updateElapsed();
}
function applySnapshot(id,data){
 if(id!==jobId)return;
 const finished=['done','error'].includes(data.status);
 const names={queued:'等待启动',running:'正在提取',done:'提取完成',error:'提取失败'};
 $('state').textContent=names[data.status]||data.status;$('state').parentElement.dataset.status=data.status;
 $('logs').textContent=data.logs||'正在启动…';renderProgress(data);downloads(id,data.files,finished);
 if(finished||['init_3d','3d','save_3d','done'].includes(data.stage))void loadTracks(id,data.files);
 if(finished){closeProgress();updateInputs(false);}
}
function connectProgress(id){
 closeProgress();startClock();updateInputs(true);
 const source=new EventSource(`/api/jobs/${id}/events`);eventStream=source;
 source.addEventListener('progress',event=>{if(id!==jobId||eventStream!==source)return;try{applySnapshot(id,JSON.parse(event.data));}catch(e){$('detail').textContent=`进度数据解析失败：${e.message}`;}});
 source.onerror=()=>{
  if(id!==jobId||eventStream!==source)return;
  $('state').textContent='进度连接中断';$('detail').textContent='正在自动重连，后台提取任务继续运行';
  if(source.readyState===EventSource.CLOSED){closeProgress();updateInputs(false);$('detail').textContent='进度连接已关闭，请重新尝试';return;}
  // A missing task after a service restart is permanent; a network interruption is retried by EventSource.
  json(`/api/jobs/${id}`).then(data=>{if(id===jobId&&eventStream===source)applySnapshot(id,data);}).catch(error=>{
   if(error.status===404&&id===jobId&&eventStream===source){closeProgress();updateInputs(false);$('state').textContent='任务已失效';$('detail').textContent='服务已重启或任务不存在，请重新生成';}
  });
 };
}
function uploadVideo(form){
 return new Promise((resolve,reject)=>{
  const xhr=new XMLHttpRequest();xhr.open('POST','/api/jobs');
  xhr.upload.onprogress=e=>{if(!e.lengthComputable)return;const percent=e.loaded/e.total*100;$('busy').value=percent;$('progressCount').hidden=false;$('progressCount').textContent=`${formatBytes(e.loaded)} / ${formatBytes(e.total)} · ${percent.toFixed(0)}%`;if(e.loaded===e.total)$('detail').textContent='上传完成，等待服务创建任务';};
  xhr.onload=()=>{try{const data=JSON.parse(xhr.responseText);if(xhr.status<200||xhr.status>=300)throw Error(data.detail||'上传失败');resolve(data);}catch(error){reject(error);}};
  xhr.onerror=()=>reject(Error('视频上传连接失败，请重试'));xhr.onabort=()=>reject(Error('视频上传已取消'));xhr.send(form);
 });
}
$('form').onsubmit=async e=>{
 e.preventDefault();const file=$('file').files[0];if(!file||file.size>512*1024*1024)return;
 resetResults();if($('video').readyState>=1)$('video').currentTime=0;
 updateInputs(true);startClock();$('busy').hidden=false;$('busy').value=0;
 $('state').textContent='上传中';$('state').parentElement.dataset.status='upload';$('detail').textContent='正在上传视频';$('logs').textContent='上传中…';
 const form=new FormData();form.append('file',file);form.append('device',$('device').value);
 try{const data=await uploadVideo(form);jobId=data.id;connectProgress(jobId);}
 catch(error){closeProgress();$('state').textContent='无法开始';$('state').parentElement.dataset.status='error';$('detail').textContent=error.message;updateInputs(false);$('busy').hidden=true;}
};
window.addEventListener('beforeunload',()=>{closeProgress();if(localVideoUrl)URL.revokeObjectURL(localVideoUrl);});
json('/api/health').then(data=>{modelReady=data.weight_ready;updateInputs(inputBusy);if(!modelReady){$('state').textContent='缺少模型';$('detail').textContent='请先放置 MHFormer model_4294.pth（详见 README）';}}).catch(e=>{$('detail').textContent=e.message;});
