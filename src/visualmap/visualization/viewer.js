'use strict';
const $ = id => document.getElementById(id);
const canvas = $('view'), ctx = canvas.getContext('2d'), video = $('sourceVideo');
const THEME = Object.freeze({
  sceneTop: '#fbfaf6',
  sceneBottom: '#f0eee5',
  grid: '#d8d3c6',
  axisX: '#b7644e',
  axisY: '#598378',
  axisZ: '#73899b',
  camera: '#477b70',
  cameraRay: '#477b7080',
  scatter: '#bc6b5045',
  pair: '#7d6c94',
  badge: '#fffefaed',
  badgeInk: '#66566f',
  labelLead: '#bc6b5080',
  labelLeadMuted: '#8f887166',
  labelText: '#794e3e',
  labelMuted: '#827b70',
  selected: '#945137',
  object: '#bc6b50',
  objectMuted: '#928b80',
  trajectory: '#ad8440a0',
  trajectoryActive: '#a27737',
  guide: '#bc6b5060',
  plan: '#f3f0e7',
  planText: '#817b70',
  videoBox: '#a9583e',
  videoLabel: '#fffefaee',
  videoText: '#563d31',
  pointRgb: '101,136,132',
});
const traj = DATA.trajectory || [], times = DATA.trajectory_timestamps || [];
const frameIds = DATA.trajectory_frame_ids || [], rotations = DATA.trajectory_rotations || [];
const finitePoint = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
const pts = (DATA.map_points || []).filter(finitePoint), objs = DATA.objects || [];
const pointRows=(DATA.map_points||[]).map((p,i)=>({p,world:DATA.map_point_world_frame_ids?.[i]})).filter(row=>finitePoint(row.p));
const submaps=DATA.summary?.submaps||[],multipleWorlds=submaps.length>1;
const poseQuality=DATA.summary?.frame_quality,hasPoseQuality=Boolean(poseQuality&&Object.keys(poseQuality).length);
let cachedWorldKey=null,cachedWorldValue=null;
function worldAtIndex(i){
  return DATA.trajectory_world_frame_ids?.[i]||DATA.summary?.frame_quality?.[String(frameIds[i])]?.world_frame_id||null;
}
function poseValid(i){
  return !hasPoseQuality||poseQuality[String(frameIds[i])]?.pose_valid===true;
}
function currentWorld(){
  const key=hasVideo?video.currentTime:tNorm;if(key===cachedWorldKey)return cachedWorldValue;
  cachedWorldKey=key;cachedWorldValue=null;
  for(let i=poseIndex();i>=0;i--){const world=worldAtIndex(i);if(world){cachedWorldValue=world;break;}}
  return cachedWorldValue;
}
function activePoints(){return multipleWorlds?pointRows.filter(row=>row.world===currentWorld()).map(row=>row.p):pts;}
function trajectoryVisible(i){return poseValid(i)&&(!multipleWorlds||worldAtIndex(i)===currentWorld());}
function trajectoryEdgeVisible(i){return i>0&&trajectoryVisible(i-1)&&trajectoryVisible(i)&&worldAtIndex(i-1)===worldAtIndex(i);}
function activeTrajectory(){return traj.filter((p,i)=>trajectoryVisible(i));}
const mapObjs = objs.filter(o => finitePoint(o.xyz) && o.depth_source !== 'default');
const observations = DATA.object_observations || [];
const correctionFingerprint=JSON.stringify([DATA.title,DATA.video_filename,DATA.summary?.video?.fps,
  DATA.trajectory?.length,objs.map(o=>[o.track_id,o.class_name,o.xyz]).slice(0,12)]);
let correctionHash=2166136261;
for(const char of correctionFingerprint)correctionHash=Math.imul(correctionHash^char.charCodeAt(0),16777619)>>>0;
const correctionKey='visualmap:human-classes:'+correctionHash.toString(16);
// Local human edits and server-owned, reversible LLM labels have separate
// lifetimes. A server label must never be persisted into the legacy local cache.
const classOverrides=new Map();
const llmClassOverrides=new Map();
const llmLabelRows=new Map();
function saveClassOverrides(){
  try{window.localStorage?.setItem(correctionKey,JSON.stringify([...classOverrides.values()]));return Boolean(window.localStorage);}catch{return false;}
}
try{
  const saved=JSON.parse(window.localStorage?.getItem(correctionKey)||'[]');
  if(Array.isArray(saved))for(const row of saved){
    if(row&&(!row.source||row.source==='user_manual_review')&&!row.label_id&&
      objs.some(o=>String(o.track_id)===String(row.track_id))&&typeof row.class_name==='string'&&row.class_name.trim()&&row.class_name.length<=40)
      classOverrides.set(String(row.track_id),row);
  }
  // Migrate already contaminated caches while preserving legitimate old edits.
  if(Array.isArray(saved)&&saved.length!==classOverrides.size)saveClassOverrides();
}catch{/* Private browsing or damaged saved notes cannot interrupt the scene. */}
const hasGeometry = observations.some(o => o.geometry);
const observationFrames = new Map();
const observationHistories = new Map();
for (const o of observations) {
  if (!observationFrames.has(o.frame_id)) observationFrames.set(o.frame_id, []);
  observationFrames.get(o.frame_id).push(o);
  if(!observationHistories.has(o.track_id))observationHistories.set(o.track_id,[]);
  observationHistories.get(o.track_id).push(o);
}
for(const rows of observationHistories.values())rows.sort((a,b)=>a.frame_id-b.frame_id);
function observationBefore(track,frame){
  const rows=observationHistories.get(track)||[];let left=0,right=rows.length;
  while(left<right){const middle=(left+right)>>>1;if(rows[middle].frame_id<frame)left=middle+1;else right=middle;}
  return rows[left-1]||null;
}
const CLASS_ZH = {person:'人',bicycle:'自行车',car:'汽车',motorcycle:'摩托车',bus:'公交车',
  truck:'卡车',bottle:'瓶子',cup:'杯子',chair:'椅子',couch:'沙发','dining table':'餐桌',
  laptop:'笔记本电脑',tv:'电视 / 显示器','cell phone':'手机',keyboard:'键盘',mouse:'鼠标',
  remote:'遥控器',backpack:'背包',book:'书',sink:'水槽',refrigerator:'冰箱',pegboard:'洞洞板（孔阵列候选）',
  perforated_panel:'带孔表面 · 类型待核对','earbud case':'耳机充电盒',earbuds:'入耳式耳机',
  headphones:'耳机','fitness tracker':'智能手环',watch:'手表 / 手环',smartwatch:'智能手表',
  tablet:'平板电脑','power adapter':'充电器','power strip':'插线板',monitor:'显示器',lamp:'台灯'};
const classLabel = name => CLASS_ZH[name] || name;
const classOverride=o=>{
  const key=String(o.track_id),manual=classOverrides.get(key),assisted=llmClassOverrides.get(key);
  // An explicit manual edit made after this particular LLM label takes priority.
  return assisted&&manual?.supersedes_label_id!==assisted.label_id?assisted:manual||assisted;
};
const displayedClass=o=>classOverride(o)?.class_name||o.class_name;
const needsClassReview=o=>o.classification_needs_review===true||o.classification_evidence?.needs_review===true;
const objectName = o => {
  const manual=classOverride(o);
  const raw=o.classification_evidence?.raw_model_class||o.classification_evidence?.original_model_class||o.class_name;
  const label=manual?manual.class_name:o.classification_evidence?.source==='open_vocabulary_model'?
    `${classLabel(o.class_name)}（待核对）`:o.classification_evidence?.source==='image_structure_rule'?
    `结构候选（${classLabel(o.class_name)}）`:needsClassReview(o)?`待核对目标（模型猜测${classLabel(raw)}）`:classLabel(o.class_name);
  return `${label} #${String(o.track_id).padStart(3,'0')}`;
};
const STATUS = {estimated:'有定位证据',tentative:'投影估计',retained:'保留历史位置',
  unavailable:'仅视频识别',moving:'移动目标估计',inconsistent:'位置与后续画面矛盾'};
const SOURCE = {sparse_observed:'本帧观测的稀疏点',sparse_projected:'稀疏点投影',
  depth_median:'局部深度中值',none:'尚无三维依据',default:'旧版默认距离',
  object_center_multiview:'多帧中心射线交会',persistent_surface_multiview:'多帧持续观测的表面点'};
const statusOf = o => STATUS[o.position_status] || (mapObjs.includes(o) ? '旧版深度估计' : '仅视频识别');
const sub = (a,b) => a.map((v,i) => v-b[i]);
const add = (a,b) => a.map((v,i) => v+b[i]);
const mul = (a,s) => a.map(v => v*s);
const dot = (a,b) => a.reduce((s,v,i) => s+v*b[i],0);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit = a => mul(a,1/(Math.hypot(...a)||1));
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const fmt = n => Number.isFinite(n) ? n.toFixed(2) : '—';
const signed = n => `${n>0?'+':''}${fmt(n)}`;
let activeClass='',selectedTrackId=null, tNorm=0, hasVideo=Boolean(DATA.video_filename);
let yaw=.62,pitch=.48,mode='perspective',follow=false,playing=false,dirty=true,clockLast=0;
const layers={trajectory:true,points:true,objects:true,volumes:true,grid:true,uncertainty:true};
let showCandidates=false;
const isSupplementCandidate=o=>o.classification_candidate_only===true||
  o.classification_evidence?.candidate_only===true||o.detection_evidence?.classification_candidate_only===true;
const candidateVisible=o=>showCandidates||Boolean(classOverride(o))||!isSupplementCandidate(o);
let currentOnly=hasGeometry,hitTargets=[],volumeHitTargets=[];
const occupancyRenderer=window.VisualMapOccupancy;
const motion = window.VisualMapMotion?.mount?.($('motionSurface'), {content:$('motionContent')}) || null;
const notifyMotion = (state,details={}) => motion?.emit(state,details);
const reducedMotion = () => typeof window.matchMedia==='function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const cameraNow = () => typeof performance!=='undefined' ? performance.now()/1000 : 0;
const cameraCanAnimate = () => typeof window.matchMedia==='function' && !reducedMotion();
let cameraUp=[0,-1,0],freeLens=null,lensMix=0,projectionMix=1,cameraMotion=null,pendingMode=null;
// Display magnification never moves the eye through the point cloud. Its wheel
// coordinate is additive, so event batching and a reverse gesture are equivalent.
let zoomInput=0;
const zoomExtent=Math.log(50);
function zoomFactor(){return Math.exp(zoomExtent*Math.tanh(Math.asinh(zoomInput)/zoomExtent));}
function zoomProjection(p,w,h){return p?{...p,x:w/2+(p.x-w/2)*zoomFactor(),y:h/2+(p.y-h/2)*zoomFactor(),scale:p.scale*zoomFactor()}:null;}
function updateZoomControls(){
  $('zoomReset').textContent=`${Math.round(zoomFactor()*100)}%`;
  $('zoomRange').value=String(Math.round(clamp(500+25*zoomInput,0,1000)));
}
const cameraHistory=[];
let lastCameraHistoryKind='',lastCameraHistoryTime=-Infinity;

function quantile(values,q){
  if(!values.length)return 0;
  const sorted=values.slice().sort((a,b)=>a-b),index=(sorted.length-1)*q,lo=Math.floor(index);
  return sorted[lo]+(sorted[Math.min(lo+1,sorted.length-1)]-sorted[lo])*(index-lo);
}
function robustCloud(points){
  if(points.length<24)return points;
  const med=[0,1,2].map(k=>quantile(points.map(p=>p[k]),.5));
  const spread=[0,1,2].map(k=>Math.max(quantile(points.map(p=>Math.abs(p[k]-med[k])),.5)*8,
    (quantile(points.map(p=>p[k]),.9)-quantile(points.map(p=>p[k]),.1))*1.5,.01));
  return points.filter(p=>p.every((v,k)=>Math.abs(v-med[k])<=spread[k]));
}
function cameraValues(){return [...target,dist,yaw,pitch,...cameraUp,lensMix,projectionMix,zoomInput];}
function cameraLimits(){return {min:Math.max(scene.radius*.002,.01),max:Math.max(scene.radius*40,1)};}
function boundedTarget(point){
  if(!finitePoint(point))return scene.center.slice();
  const offset=sub(point,scene.center),length=Math.hypot(...offset),limit=scene.radius*120;
  return length>limit?add(scene.center,mul(offset,limit/length)):point.slice();
}
function applyCameraValues(values){
  const limits=cameraLimits();target=boundedTarget(values.slice(0,3));
  dist=Number.isFinite(values[3])?clamp(values[3],limits.min,limits.max):scene.radius*3.2;
  yaw=Number.isFinite(values[4])?values[4]:.62;pitch=Number.isFinite(values[5])?clamp(values[5],-Math.PI/2+.0001,Math.PI/2-.0001):.48;
  const up=values.slice(6,9);cameraUp=finitePoint(up)&&Math.hypot(...up)>.001?unit(up):[0,-1,0];
  lensMix=Number.isFinite(values[9])?clamp(values[9],0,1):0;
  projectionMix=Number.isFinite(values[10])?clamp(values[10],0,1):1;
  zoomInput=Number.isFinite(values[11])?values[11]:0;
}
function sampleCameraMotion(now=cameraNow()){
  if(!cameraMotion)return;
  if(cameraMotion.finishMode==='camera'){
    const pose=poseAt();
    if(pose.R){const forward=pose.R.map(row=>row[2]),up=pose.R.map(row=>-row[1]);
      cameraMotion.goal.splice(0,3,...add(pose.p,mul(forward,cameraMotion.goal[3])));
      cameraMotion.goal[4]=cameraMotion.start[4]+Math.atan2(Math.sin(Math.atan2(-forward[0],forward[2])-cameraMotion.start[4]),Math.cos(Math.atan2(-forward[0],forward[2])-cameraMotion.start[4]));
      cameraMotion.goal[5]=Math.asin(clamp(forward[1],-1,1));cameraMotion.goal.splice(6,3,...up);
    }
  }
  const time=Math.max(0,now-cameraMotion.started),omega=12,decay=Math.exp(-omega*time);
  const values=[],velocities=[];
  cameraMotion.goal.forEach((goal,i)=>{
    const delta=cameraMotion.start[i]-goal,c=cameraMotion.velocity[i]+omega*delta;
    values.push(goal+(delta+c*time)*decay);
    velocities.push((cameraMotion.velocity[i]-omega*c*time)*decay);
  });
  const fromUp=unit(cameraMotion.start.slice(6,9)),toUp=unit(cameraMotion.goal.slice(6,9)),upDot=dot(fromUp,toUp);
  if(upDot<-.95){
    let perpendicular=unit(sub(toUp,mul(fromUp,upDot)));
    if(Math.hypot(...perpendicular)<.1)perpendicular=unit(cross(fromUp,Math.abs(fromUp[0])<.8?[1,0,0]:[0,0,1]));
    const angle=Math.acos(clamp(upDot,-1,1))*(1-(1+omega*time)*decay);
    values.splice(6,3,...add(mul(fromUp,Math.cos(angle)),mul(perpendicular,Math.sin(angle))));
  }
  applyCameraValues(values);cameraMotion.currentVelocity=velocities;dirty=true;
  if(time>=.85){
    const finalMode=cameraMotion.finishMode;applyCameraValues(cameraMotion.goal);cameraMotion=null;
    if(finalMode){mode=finalMode;pendingMode=null;syncCameraControls();}
  }
}
function stopCameraMotion(){sampleCameraMotion();cameraMotion=null;}
function moveCamera(goal,clearVelocity=false){
  const now=cameraNow();sampleCameraMotion(now);
  applyCameraValues(cameraValues());
  const start=cameraValues(),velocity=!clearVelocity&&cameraMotion?.currentVelocity||start.map(()=>0);
  const values=[...(goal.target||target),goal.dist??dist,goal.yaw??yaw,goal.pitch??pitch,
    ...(goal.up||cameraUp),goal.lensMix??lensMix,goal.projectionMix??projectionMix,goal.zoomInput??zoomInput];
  // Take the nearest yaw turn rather than spinning through the world origin.
  values[4]=start[4]+Math.atan2(Math.sin(values[4]-start[4]),Math.cos(values[4]-start[4]));
  if(cameraCanAnimate())cameraMotion={start,goal:values,velocity,started:now,currentVelocity:velocity};
  else{cameraMotion=null;applyCameraValues(values);}
  dirty=true;
}
function rememberCamera(kind='navigation'){
  const now=cameraNow();sampleCameraMotion(now);
  if(kind==='wheel'&&lastCameraHistoryKind===kind&&now-lastCameraHistoryTime<.35){lastCameraHistoryTime=now;return;}
  const snapshot={values:cameraValues(),mode,pendingMode,follow,freeLens};
  const previous=cameraHistory[cameraHistory.length-1];
  if(!previous||JSON.stringify(snapshot)!==JSON.stringify(previous)){
    cameraHistory.push(snapshot);if(cameraHistory.length>40)cameraHistory.shift();
  }
  lastCameraHistoryKind=kind;lastCameraHistoryTime=now;
}
function undoCamera(){
  const snapshot=cameraHistory.pop();if(!snapshot){notifyMotion('toast',{label:'已经是最早的视角'});return;}
  cancelCameraDrag();cameraMotion=null;pendingMode=null;freeLens=snapshot.freeLens;
  applyCameraValues(snapshot.values);mode=snapshot.pendingMode||snapshot.mode;follow=snapshot.follow;
  lastCameraHistoryKind='';syncCameraControls();dirty=true;notifyMotion('view',{label:'返回上一个视角'});
}
function viewerBasis(){
  const pose=poseAt();
  return mode==='camera'&&pose.R?{eye:pose.p,x:pose.R.map(row=>row[0]),
    y:pose.R.map(row=>-row[1]),z:pose.R.map(row=>row[2])}:lookAtBasis(orbitEye(),target,cameraUp);
}
function syncCameraControls(){
  $('followBtn').textContent=follow?'跟随相机 · 开':'跟随相机';
  $('followBtn').classList.toggle('active',follow);
  $('followBtn').setAttribute('aria-pressed',String(follow));
  updateViewButtons();
}
function leaveCameraMode(){
  const interrupted=Boolean(cameraMotion||pendingMode);
  const requestedProjection=cameraMotion?.goal[10]??projectionMix;
  stopCameraMotion();
  pendingMode=null;
  if(mode==='camera'){
    const basis=viewerBasis(),depths=visibleMapObjects().map(o=>dot(sub(currentPosition(o),basis.eye),basis.z)).filter(d=>d>.01);
    const focusDepth=depths.length?quantile(depths,.5):Math.max(scene.radius*.6,.5);
    target=add(basis.eye,mul(basis.z,focusDepth));dist=focusDepth;
    yaw=Math.atan2(-basis.z[0],basis.z[2]);pitch=Math.asin(clamp(basis.z[1],-1,1));
    cameraUp=basis.y.slice();freeLens=DATA.summary?.intrinsics||null;lensMix=freeLens?1:0;
    mode='perspective';projectionMix=1;
  }else if(interrupted){
    const nextProjection=requestedProjection<.5?0:1;
    dist*=(nextProjection?1.15:1.5)/(projectionMix*1.15+(1-projectionMix)*1.5);
    projectionMix=nextProjection;cameraUp=[0,-1,0];
    mode=nextProjection?'perspective':'orthographic';
  }
  follow=false;syncCameraControls();dirty=true;
}
function normalizeNavigationLens(){
  // A calibrated, distorted video lens is useful for evidence, but not for free orbit.
  // Keep the focus plane's apparent position and size when leaving that lens.
  if(!freeLens||lensMix<1e-5){freeLens=null;lensMix=0;return;}
  const basis=viewerBasis(),w=canvas.clientWidth,h=canvas.clientHeight;
  const screen=project(target,basis,w,h),limits=cameraLimits(),focal=Math.min(w,h)*1.15;
  if(screen&&Number.isFinite(screen.scale)&&screen.scale>1e-8){
    dist=clamp(focal*zoomFactor()/screen.scale,limits.min,limits.max);
    const scale=focal/dist*zoomFactor();
    target=boundedTarget(add(target,add(mul(basis.x,(w/2-screen.x)/scale),mul(basis.y,(screen.y-h/2)/scale))));
  }
  freeLens=null;lensMix=0;projectionMix=1;
}
function fittedScenePoints(){
  const positioned=visibleMapObjects().filter(o=>positionSnapshot(o).position_temporal_consistency!=='contradicted').map(currentPosition);
  return [...activeTrajectory(),...robustCloud(activePoints()),...positioned,...visibleOccupancies().flatMap(item=>occupancyRenderer?.corners(item.bounds)||[])];
}

function bounds(points) {
  const all=points.filter(finitePoint);
  if (!all.length) return {center:[0,0,0],radius:3,min:[-2,-2,-2],max:[2,2,2]};
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
  for (const p of all) for(let k=0;k<3;k++){min[k]=Math.min(min[k],p[k]);max[k]=Math.max(max[k],p[k]);}
  return {min,max,center:min.map((v,k)=>(v+max[k])/2),radius:Math.max(.5,Math.hypot(...sub(max,min))/2)};
}
let scene=bounds(multipleWorlds?[...activeTrajectory(),...robustCloud(activePoints())]:[...traj,...robustCloud(pts),...mapObjs.filter(o=>o.position_temporal_consistency!=='contradicted').map(o=>o.xyz)]);
let planScene=multipleWorlds?scene:bounds([...traj,...mapObjs.map(o=>o.xyz)]);
const initialScene=multipleWorlds?scene:bounds([...traj,...mapObjs.map(o=>o.xyz)].length?
  [...traj,...mapObjs.map(o=>o.xyz)]:pts);
let displayedWorld=worldAtIndex(0);
let target=initialScene.center.slice(),dist=initialScene.radius*2.8;
const allObjects=()=>objs.filter(o=>candidateVisible(o)&&(!currentOnly||o.manual_llm_object||(observationFrames.get(sourceFrame())||[])
  .some(d=>d.track_id===o.track_id))&&(!activeClass||displayedClass(positionSnapshot(o))===activeClass));
function poseIndex(){
  const fps=DATA.summary?.video?.fps;
  if(hasVideo&&Number.isFinite(fps)&&fps>0){
    const frame=Math.floor(video.currentTime*fps+1e-4),i=indexForFrame(frame);
    if(frameIds[i]===frame)return i;
  }
  return Math.floor(tNorm*Math.max(0,traj.length-1)+1e-8);
}
function currentFrame(){return frameIds[poseIndex()] ?? 0;}
function sourceFrame(){
  const fps=DATA.summary?.video?.fps;
  // Browser times and CSV timestamps may be rounded to microseconds.
  return hasVideo&&Number.isFinite(fps)&&fps>0?Math.floor(video.currentTime*fps+1e-4):currentFrame();
}
function poseAt(){
  const i=poseIndex();
  return {p:traj[i]||[0,0,0],R:rotations[i]||null};
}
function positionSnapshot(o,forceCurrent=false){
  if (!currentOnly&&!forceCurrent){
    const g=historicalOccupancyGeometry(o);
    if(!g)return o;
    const row=(observationFrames.get(g.frame_id)||[]).find(d=>d.track_id===o.track_id)||{};
    return {...o,...row,xyz:g.xyz,geometry:g,occupancy:g.occupancy,frame_id:g.frame_id,
      position_frame:g.position_frame,position_source:g.position_source,
      position_status:g.frame_id<sourceFrame()?'retained':g.status,
      position_uncertainty:g.uncertainty,position_temporal_consistency:g.temporal_consistency,
      position_relation_usable:false,world_frame_id:g.world_frame_id};
  }
  const observation=(observationFrames.get(sourceFrame())||[]).find(d=>d.track_id===o.track_id);
  const history=(o.position_history||[]).filter(p=>p.frame_id<=sourceFrame()&&finitePoint(p.xyz));
  const p=history[history.length-1];
  const classification=observation?{frame_id:observation.frame_id,xyxy:observation.xyxy,
    class_name:observation.class_name,confidence:observation.confidence,
    observations:observation.hits,
    classification_consistency:observation.class_consistency,
    classification_observations:observation.class_observations,
    classification_needs_review:observation.classification_needs_review,
    classification_reason:observation.classification_reason,
    detection_evidence:observation.detection_evidence,
    classification_evidence:{...(observation.classification_evidence||{}),latest_raw_class:observation.raw_class_name,
      alternatives:observation.class_alternatives||[]}}:{};
  if(observation?.geometry){
    const g=observation.geometry;
    return {...o,...classification,observation_current:true,xyz:g.xyz,position_frame:g.position_frame,
      position_source:g.position_source,position_uncertainty:g.uncertainty,
      position_status:g.status,position_temporal_consistency:g.temporal_consistency,
      position_relation_usable:g.position_relation_usable===true,
      position_reprojection:g.reprojection,evidence:g.support_evidence||g.multiview_evidence||{},
      geometry:g,occupancy:g.occupancy,world_frame_id:g.world_frame_id};
  }
  if(!observation||!p){
    const past=observation?null:observationBefore(o.track_id,sourceFrame());
    const historical=!observation?{frame_id:past?.frame_id??null,xyxy:null,
      class_name:past?.class_name||'未观测目标',confidence:past?.confidence??null,observations:past?.hits??0,
      classification_consistency:past?.class_consistency,classification_observations:past?.class_observations,
      classification_needs_review:past?.classification_needs_review,classification_reason:past?.classification_reason,
      classification_evidence:past?.classification_evidence||{},detection_evidence:past?.detection_evidence}:{};
    return {...o,...classification,...historical,observation_current:Boolean(observation),xyz:null,
      position_frame:null,position_uncertainty:null,position_status:'unavailable',position_observations:0,
      position_source:'none',depth_source:'none',position_temporal_consistency:'unverified',
      position_relation_usable:false,position_reprojection:null,evidence:{},geometry:null,occupancy:null,world_frame_id:null};
  }
  const snapshot={...o,...classification,observation_current:true,xyz:p.xyz,position_frame:p.frame_id,position_uncertainty:p.uncertainty,
    position_status:p.frame_id===sourceFrame()?p.status:'retained',position_source:p.source,
    position_observations:history.length,evidence:{},position_temporal_consistency:p.position_temporal_consistency||'unverified'};
  const projection=reprojectToObservation(p.xyz,observation,o.position_reprojection?.box_margin??.15);
  if(projection){
    snapshot.position_reprojection=projection;
    snapshot.position_temporal_consistency=projection.inside_box?'consistent':'contradicted';
    if(!projection.inside_box)snapshot.position_status='inconsistent';
  }
  return snapshot;
}
const currentPosition=o=>positionSnapshot(o).xyz;
function reprojectToObservation(point,observation,margin){
  const i=indexForFrame(observation.frame_id),R=rotations[i],intr=DATA.summary?.intrinsics;
  if(!R||!intr||!observation.xyxy||frameIds[i]!==observation.frame_id)return null;
  const delta=sub(point,traj[i]),p=[0,1,2].map(k=>R.reduce((sum,row,j)=>sum+row[k]*delta[j],0));
  if(p[2]<=0)return {frame_id:observation.frame_id,pixel:null,inside_box:false};
  const x=p[0]/p[2],y=p[1]/p[2],r2=x*x+y*y,[k1=0,k2=0,p1=0,p2=0,k3=0]=intr.dist||[];
  const radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2;
  const u=intr.fx*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+intr.cx;
  const v=intr.fy*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+intr.cy;
  const [x1,y1,x2,y2]=observation.xyxy,mx=(x2-x1)*margin,my=(y2-y1)*margin;
  const inside=Number.isFinite(u)&&Number.isFinite(v)&&u>=0&&u<intr.width&&v>=0&&v<intr.height&&
    u>=x1-mx&&u<=x2+mx&&v>=y1-my&&v<=y2+my;
  return {frame_id:observation.frame_id,pixel:[u,v],inside_box:inside};
}
const visibleMapObjects=()=>mapObjs.filter(o=>{
  if(!candidateVisible(o))return false;
  const snapshot=positionSnapshot(o),world=snapshot.world_frame_id||snapshot.geometry?.world_frame_id;
  return (!multipleWorlds||world===currentWorld())&&(!activeClass||displayedClass(snapshot)===activeClass)&&snapshot.xyz;
});
function historicalOccupancyGeometry(o){
  return (o.position_snapshots||[]).filter(g=>g.frame_id<=sourceFrame()&&g.occupancy?.available===true&&(!multipleWorlds||g.world_frame_id===currentWorld()))
    .sort((a,b)=>b.frame_id-a.frame_id)[0]||null;
}
function occupancyFor(o,forceCurrent=false){
  const current=currentOnly||forceCurrent,snapshot=positionSnapshot(o,forceCurrent);
  const geometry=current?snapshot.geometry:historicalOccupancyGeometry(o);
  const volume=geometry?.occupancy;
  if(!volume||volume.available!==true||!occupancyRenderer?.validBounds(volume))return null;
  if(multipleWorlds&&volume.world_frame_id!==currentWorld())return null;
  if(volume.frame_id!==geometry.frame_id||volume.frame_id>sourceFrame()||
    geometry.pose_valid!==true||geometry.pose_trusted!==true||!volume.world_frame_id||
    volume.world_frame_id!==geometry.world_frame_id)return null;
  if(current){
    const geometry=snapshot.geometry,quality=DATA.summary?.frame_quality?.[String(sourceFrame())];
    if(!geometry||volume.frame_id!==sourceFrame()||geometry.frame_id!==sourceFrame()||
      quality?.pose_valid!==true||geometry.pose_valid!==true||!volume.world_frame_id||
      volume.world_frame_id!==geometry.world_frame_id)return null;
  }
  return volume;
}
function visibleOccupancies(){
  return objs.filter(o=>candidateVisible(o)&&(!activeClass||displayedClass(positionSnapshot(o))===activeClass)).map(o=>{
    const volume=occupancyFor(o);return volume?{o,trackId:o.track_id,volume,
      bounds:{...volume,world_id:volume.world_frame_id},extent_kind:'inferred'}:null;
  }).filter(Boolean);
}
function occupancyCorners(o,forceCurrent=false){
  const volume=occupancyFor(o,forceCurrent);return volume?occupancyRenderer.corners(volume):[];
}
function occupancyPairState(a,b){
  const unavailable=reason=>({usable:false,reason});
  if(!a||!b||a===b)return unavailable('请选择上方两个不同的物体。');
  if(DATA.summary?.scale_consistent!==true)return unavailable('本段视频尚未建立一致的空间尺度。');
  const va=occupancyFor(a,true),vb=occupancyFor(b,true);
  if(!va||!vb)return unavailable('当前帧没有两处可用的体积证据，碰撞预估为未知。');
  if(va.candidate_overlap_usable!==true||vb.candidate_overlap_usable!==true)
    return unavailable('物体归属或定位证据仍待核对，碰撞预估为未知。');
  const quality=DATA.summary?.frame_quality?.[String(sourceFrame())],poseIndex=indexForFrame(sourceFrame());
  const world=DATA.trajectory_world_frame_ids?.[poseIndex]||quality?.world_frame_id||DATA.summary?.world_frame_id;
  if(frameIds[poseIndex]!==sourceFrame()||!world||world!==va.world_frame_id||world!==vb.world_frame_id)
    return unavailable('两处体积与当前相机不在同一坐标系，暂不比较。');
  const margin=Number($('occupancyMargin').value);
  if(!Number.isFinite(margin)||margin<0)return unavailable('预留间隙请输入不小于0的数字。');
  return {...occupancyRenderer.potentialOverlap({...va,world_id:va.world_frame_id},{...vb,world_id:vb.world_frame_id},{margin}),a:va,b:vb};
}
function updateOccupancyReadout(){
  const el=$('occupancyReadout');el.replaceChildren();const [a,b]=pairObjects(),result=occupancyPairState(a,b);
  const message=document.createElement('strong');
  message.textContent=!result.usable?result.reason:result.overlap?
    (result.raw_overlap?'估计占用体存在潜在交叠':'加入预留间隙后存在潜在交叠'):'当前估计占用体暂未交叠';
  message.className=result.usable&&result.overlap?'occupancy-overlap':'occupancy-status';el.append(message);
  if(!result.usable)return;
  const metrics=document.createElement('div');metrics.className='metrics occupancy-metrics';
  for(const [label,value] of [['物体A · X / Y / Z',result.a.size.map(fmt).join(' / ')],
    ['物体B · X / Y / Z',result.b.size.map(fmt).join(' / ')],['包围体间隙',fmt(result.gap)+' units'],
    ['预留间隙',fmt(result.margin)+' units / 物体']]){
    const item=document.createElement('div'),small=document.createElement('small'),big=document.createElement('b');
    item.className='metric';small.textContent=label;big.textContent=value;item.append(small,big);metrics.append(item);
  }
  el.append(metrics);
  const note=document.createElement('p');note.className='micro';
  note.textContent='此处比较当前帧的估计包围体。多个跟踪记录可能属于同一物体；未观测背面和边界误差仍待核对，未交叠不能作为安全通行结论。';el.append(note);
}
function normToTime(u){
  if(times.length===traj.length&&times.length){
    const x=clamp(u,0,1)*(times.length-1),i=Math.floor(x),j=Math.min(i+1,times.length-1);
    return times[i]+(times[j]-times[i])*(x-i);
  }
  return u*(Number.isFinite(video.duration)?video.duration:0);
}
function timeToNorm(seconds){
  if(times.length!==traj.length||times.length<2) return video.duration?clamp(seconds/video.duration,0,1):0;
  let lo=0,hi=times.length-1;
  if(seconds<=times[lo])return 0;if(seconds>=times[hi])return 1;
  while(hi-lo>1){const mid=(lo+hi)>>1;if(times[mid]<=seconds)lo=mid;else hi=mid;}
  return (lo+(seconds-times[lo])/(times[hi]-times[lo]||1))/(times.length-1);
}
function indexForFrame(frame){
  if(!frameIds.length)return 0;
  let lo=0,hi=frameIds.length-1;
  while(hi-lo>1){const mid=(lo+hi)>>1;if(frameIds[mid]<=frame)lo=mid;else hi=mid;}
  return Math.abs(frameIds[lo]-frame)<=Math.abs(frameIds[hi]-frame)?lo:hi;
}
const timeForFrame=frame=>times[indexForFrame(frame)] ?? 0;
function lookAtBasis(eye,center,up=[0,-1,0]){
  const z=unit(sub(center,eye));
  let x=unit(cross(z,up));
  if(Math.hypot(...x)<.1)x=[1,0,0];
  return {x,y:unit(cross(x,z)),z,eye};
}
function orbitEye(){
  return add(target,[dist*Math.cos(pitch)*Math.sin(yaw),-dist*Math.sin(pitch),-dist*Math.cos(pitch)*Math.cos(yaw)]);
}
function project(p,basis,w,h){
  if(!finitePoint(p))return null;
  const v=sub(p,basis.eye),depth=dot(v,basis.z);
  if(depth<=Math.max(.001,scene.radius*.0001))return null;
  const activeLens=mode==='camera'?DATA.summary?.intrinsics:freeLens;
  const blend=mode==='camera'?1:lensMix;
  if(activeLens&&blend>1e-5){
    const intr=activeLens,x=dot(v,basis.x)/depth,y=-dot(v,basis.y)/depth;
    const r2=x*x+y*y,[k1=0,k2=0,p1=0,p2=0,k3=0]=intr.dist||[];
    const radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2,fitScale=Math.min(w/intr.width,h/intr.height);
    const u=intr.fx*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+intr.cx;
    const vv=intr.fy*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+intr.cy;
    const calibrated={x:(w-intr.width*fitScale)/2+u*fitScale,y:(h-intr.height*fitScale)/2+vv*fitScale,
      z:depth,scale:intr.fx*fitScale/depth};
    if(blend>=.99999)return zoomProjection(calibrated,w,h);
    const scale=(Math.min(w,h)*1.15/depth)*projectionMix+(Math.min(w,h)*1.5/dist)*(1-projectionMix);
    return zoomProjection({x:(w/2+dot(v,basis.x)*scale)*(1-blend)+calibrated.x*blend,
      y:(h/2-dot(v,basis.y)*scale)*(1-blend)+calibrated.y*blend,z:depth,
      scale:scale*(1-blend)+calibrated.scale*blend},w,h);
  }
  const scale=(Math.min(w,h)*1.15/depth)*projectionMix+(Math.min(w,h)*1.5/dist)*(1-projectionMix);
  return zoomProjection({x:w/2+dot(v,basis.x)*scale,y:h/2-dot(v,basis.y)*scale,z:depth,scale},w,h);
}
function fittedDistance(points,center,angles,orthographic=false){
  const w=Math.max(canvas.clientWidth,1),h=Math.max(canvas.clientHeight,1),focal=Math.min(w,h)*(orthographic?1.5:1.15);
  const direction=[-Math.cos(angles.pitch)*Math.sin(angles.yaw),Math.sin(angles.pitch),Math.cos(angles.pitch)*Math.cos(angles.yaw)];
  const basis=lookAtBasis(sub(center,direction),center,angles.up||cameraUp);
  let distance=.5;
  for(const point of points.filter(finitePoint)){
    const v=sub(point,center),depth=orthographic?0:dot(v,basis.z);
    distance=Math.max(distance,Math.abs(dot(v,basis.x))*focal/(w*.42)-depth,
      Math.abs(dot(v,basis.y))*focal/(h*.42)-depth,-depth+.01);
  }
  return clamp(distance,cameraLimits().min,cameraLimits().max);
}
function fit(points,full=false){
  rememberCamera('fit');cancelCameraDrag();leaveCameraMode();
  // The cloud was filtered separately. Filtering the combined cloud again can discard
  // valid camera positions or objects simply because there are fewer of them.
  const box=bounds(points),angles=full?{yaw:.62,pitch:.48,up:[0,-1,0]}:{yaw,pitch,up:cameraUp};
  if(full){
    normalizeNavigationLens();
    dist*=1.15/(projectionMix*1.15+(1-projectionMix)*1.5);
    projectionMix=1;mode='perspective';
  }
  freeLens=null;
  moveCamera({target:box.center,dist:fittedDistance(points,box.center,angles,!full&&projectionMix===0),lensMix:0,projectionMix:full?1:projectionMix,
    zoomInput:0,...angles},true);
  syncCameraControls();
  notifyMotion('view',{label:full?'全景':'聚焦',mode});
  dirty=true;
}
function setView(next){
  rememberCamera('view');cancelCameraDrag();
  if(next==='camera'){
    const pose=poseAt();if(!pose.R){notifyMotion('toast',{label:'本帧没有镜头姿态'});return;}
    stopCameraMotion();const forward=pose.R.map(row=>row[2]),up=pose.R.map(row=>-row[1]);
    const depths=visibleMapObjects().map(o=>dot(sub(currentPosition(o),pose.p),forward)).filter(d=>d>.01);
    const focusDepth=depths.length?quantile(depths,.5):Math.max(scene.radius*.6,.5);
    // A source-camera view is an exact pose, not an interruptible blended lens.
    target=add(pose.p,mul(forward,focusDepth));dist=focusDepth;
    yaw=Math.atan2(-forward[0],forward[2]);pitch=Math.asin(clamp(forward[1],-1,1));
    cameraUp=up;freeLens=null;lensMix=0;projectionMix=1;zoomInput=0;
    mode='camera';pendingMode=null;
    follow=false;syncCameraControls();dirty=true;
  }else{
    leaveCameraMode();normalizeNavigationLens();mode=next;
    freeLens=null;
    const angles=next==='top'?{pitch:Math.PI/2-.0001,yaw:0}:next==='front'?{pitch:0,yaw:0}:
      next==='side'?{pitch:0,yaw:Math.PI/2}:{pitch:.48,yaw:.62};
    const nextProjection=next==='perspective'?1:0;
    // Direction buttons give the same useful framing after arbitrary navigation.
    // Change projection explicitly here, preserving the focus-plane scale at entry;
    // pointer release never changes projection or magnification.
    dist*=(nextProjection?1.15:1.5)/(projectionMix*1.15+(1-projectionMix)*1.5);
    projectionMix=nextProjection;
    const points=fittedScenePoints(),box=bounds(points);
    moveCamera({...angles,target:box.center,dist:fittedDistance(points,box.center,{...angles,up:[0,-1,0]},!nextProjection),
      up:[0,-1,0],lensMix:0,zoomInput:0},true);syncCameraControls();
  }
  notifyMotion('view',{label:{perspective:'立体视角',top:'顶视',front:'正视',side:'侧视',camera:'镜头视角'}[next],mode:next});
}
function updateViewButtons(){
  const displayedMode=pendingMode||mode;
  for(const id of ['perspective','top','front','side','camera']){
    $(id+'Btn').classList.toggle('active',id===displayedMode);
    $(id+'Btn').setAttribute('aria-pressed',String(id===displayedMode));
  }
  $('viewHud').textContent={perspective:'三维透视',orthographic:'自由旋转 · 正交',top:'顶视 · X / Z',front:'正视 · X / Y',side:'侧视 · Z / Y',camera:'沿视频镜头查看'}[displayedMode];
}
function resizeCanvas(el){
  const dpr=Math.min(window.devicePixelRatio||1,2),w=el.clientWidth,h=el.clientHeight;
  if(el.width!==Math.round(w*dpr)||el.height!==Math.round(h*dpr)){
    el.width=Math.round(w*dpr);el.height=Math.round(h*dpr);
  }
  const context=el.getContext('2d');context.setTransform(dpr,0,0,dpr,0,0);return context;
}
function line(a,b,basis,w,h,color,width=1,dash=[]){
  const p=project(a,basis,w,h),q=project(b,basis,w,h);if(!p||!q)return;
  ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();
  ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();ctx.setLineDash([]);
}
function drawGrid(basis,w,h){
  if(!layers.grid)return;
  const span=Math.max(scene.max[0]-scene.min[0],scene.max[2]-scene.min[2],1);
  const base=10**Math.floor(Math.log10(span/10)),step=Math.ceil(span/(10*base))*base;
  const x0=Math.floor(scene.min[0]/step)*step,x1=Math.ceil(scene.max[0]/step)*step;
  const z0=Math.floor(scene.min[2]/step)*step,z1=Math.ceil(scene.max[2]/step)*step,y=scene.max[1];
  for(let x=x0;x<=x1+step*.1;x+=step)line([x,y,z0],[x,y,z1],basis,w,h,THEME.grid);
  for(let z=z0;z<=z1+step*.1;z+=step)line([x0,y,z],[x1,y,z],basis,w,h,THEME.grid);
  $('gridLabel').textContent=`网格 ${fmt(step)} map units · 非地面`;
}
function drawAxes(basis,w,h){
  // Fixed screen compass uses the same basis as scene, without moving the origin.
  const origin={x:55,y:h-65},len=28;
  for(const [axis,color,label] of [[[1,0,0],THEME.axisX,'X'],[[0,1,0],THEME.axisY,'Y'],[[0,0,1],THEME.axisZ,'Z']]){
    const x=origin.x+dot(axis,basis.x)*len,y=origin.y-dot(axis,basis.y)*len;
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(origin.x,origin.y);ctx.lineTo(x,y);ctx.stroke();
    ctx.fillStyle=color;ctx.font='11px Geist, "Microsoft YaHei", sans-serif';ctx.fillText(label,x+4,y-3);
  }
}
function drawFrustum(pose,basis,w,h){
  const screen=project(pose.p,basis,w,h);if(!screen)return;
  ctx.fillStyle=THEME.camera;ctx.beginPath();ctx.arc(screen.x,screen.y,5,0,Math.PI*2);ctx.fill();
  if(!pose.R)return;
  const intr=(DATA.summary||{}).intrinsics||{},depth=scene.radius*.10;
  const x0=intr.fx?-intr.cx/intr.fx:-.5,x1=intr.fx?(intr.width-intr.cx)/intr.fx:.5;
  const y0=intr.fy?-intr.cy/intr.fy:-.36,y1=intr.fy?(intr.height-intr.cy)/intr.fy:.36;
  const transform=p=>add(pose.p,pose.R.map(row=>dot(row,p)));
  const corners=[[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]].map(p=>transform(mul(p,depth)));
  corners.forEach((p,i)=>{line(pose.p,p,basis,w,h,THEME.cameraRay,1.5);line(p,corners[(i+1)%4],basis,w,h,THEME.camera,1.5);});
  line(pose.p,transform([0,0,depth*1.4]),basis,w,h,THEME.camera,2);
}
function drawScatter(o,p,basis,w,h){
  const r=o.position_uncertainty;if(!layers.uncertainty||!Number.isFinite(r)||r<=0)return;
  for(const [a,b] of [[0,1],[0,2],[1,2]]){
    let previous=null;
    for(let i=0;i<=32;i++){
      const q=p.slice();q[a]+=r*Math.cos(i*Math.PI/16);q[b]+=r*Math.sin(i*Math.PI/16);
      if(previous)line(previous,q,basis,w,h,THEME.scatter,1,[3,3]);previous=q;
    }
  }
}
function pairObjects(){
  return [$('objectA').value,$('objectB').value].map(id=>objs.find(o=>String(o.track_id)===id));
}
function drawPair(basis,w,h){
  const [a,b]=pairObjects();if(!a||!b||a===b||!layers.objects)return;
  const relation=relationState(a,b);
  if(!relation.usable)return;
  const pa=relation.a.xyz,pb=relation.b.xyz;
  line(pa,pb,basis,w,h,THEME.pair,2,[5,4]);
  const p=project(mul(add(pa,pb),.5),basis,w,h);if(!p)return;
  ctx.font='12px Geist, "Microsoft YaHei", sans-serif';const label=`${fmt(Math.hypot(...sub(pb,pa)))} units`;
  const tw=ctx.measureText(label).width;
  ctx.fillStyle=THEME.badge;ctx.fillRect(p.x-tw/2-6,p.y-19,tw+12,22);
  ctx.fillStyle=THEME.badgeInk;ctx.fillText(label,p.x-tw/2,p.y-4);
}
function drawLabels(labels,w,h){
  const occupied=[];
  labels.sort((a,b)=>Number(b.selected)-Number(a.selected)||a.s.z-b.s.z);
  for(const {o,s,selected,conflict} of labels){
    const label=objectName(o)+(conflict?' · 待复核':'');
    ctx.font='12px Geist, "Microsoft YaHei", sans-serif';const width=ctx.measureText(label).width+12,height=22;
    let box=null;
    for(const offset of [0,-26,26,-52,52,-78,78,-104,104,-130,130]){
      for(const direction of [1,-1]){
        const candidate={x:direction===1?s.x+18:s.x-18-width,y:s.y-11+offset,width,height};
        if(candidate.x<8||candidate.x+width>w-8||candidate.y<62||candidate.y+height>h-32)continue;
        if(occupied.some(b=>candidate.x<b.x+b.width+5&&candidate.x+width+5>b.x&&
          candidate.y<b.y+b.height+4&&candidate.y+height+4>b.y))continue;
        box=candidate;break;
      }
      if(box)break;
    }
    if(!box)continue;
    occupied.push(box);ctx.strokeStyle=conflict?THEME.labelLeadMuted:THEME.labelLead;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(s.x,s.y);ctx.lineTo(box.x+(box.x>s.x?0:box.width),box.y+height/2);ctx.stroke();
    ctx.fillStyle=THEME.badge;ctx.fillRect(box.x,box.y,width,height);
    ctx.fillStyle=selected?THEME.selected:conflict?THEME.labelMuted:THEME.labelText;ctx.fillText(label,box.x+6,box.y+15);
  }
}
function draw(){
  resizeCanvas(canvas);const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;
  const bg=ctx.createLinearGradient(0,0,0,h);bg.addColorStop(0,THEME.sceneTop);bg.addColorStop(1,THEME.sceneBottom);
  ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
  const pose=poseAt(),basis=viewerBasis();
  drawGrid(basis,w,h);
  if(layers.points){
    const projected=activePoints().map(p=>project(p,basis,w,h)).filter(Boolean).sort((a,b)=>b.z-a.z);
    for(const s of projected){
      ctx.fillStyle=`rgba(${THEME.pointRgb},${clamp(.7-s.z/(scene.radius*12),.15,.7)})`;
      ctx.beginPath();ctx.arc(s.x,s.y,clamp(s.scale*.018,1,2.6),0,Math.PI*2);ctx.fill();
    }
  }
  if(layers.trajectory){
    for(let i=1;i<traj.length;i++)if(trajectoryEdgeVisible(i))line(traj[i-1],traj[i],basis,w,h,THEME.trajectory,1.8);
    const index=poseIndex();
    for(let i=Math.max(1,index-8);i<=index;i++)if(trajectoryEdgeVisible(i))line(traj[i-1],traj[i],basis,w,h,THEME.trajectoryActive,3);
  }
  hitTargets=[];
  volumeHitTargets=[];
  if(layers.objects&&layers.volumes&&occupancyRenderer){
    const rendered=occupancyRenderer.render(ctx,visibleOccupancies(),{basis,
      project:p=>project(p,basis,w,h),width:w,height:h,near:Math.max(.001,scene.radius*.0001)*1.001,
      selectedTrackId,orthographic:projectionMix===0});
    volumeHitTargets=rendered.hitTargets;
  }
  if(layers.objects){
    const labels=[];
    const positions=visibleMapObjects().map(o=>({o,p:currentPosition(o)}));
    positions.sort((a,b)=>(project(b.p,basis,w,h)?.z||0)-(project(a.p,basis,w,h)?.z||0));
    for(const {o,p} of positions){
      const s=project(p,basis,w,h);if(!s)continue;
      if(o.track_id===selectedTrackId)drawScatter(positionSnapshot(o),p,basis,w,h);
      // Guide to reference plane separates elevation from depth.
      line(p,[p[0],scene.max[1],p[2]],basis,w,h,THEME.guide,1,[3,4]);
      const snapshot=positionSnapshot(o),conflict=snapshot.position_temporal_consistency==='contradicted';
      const selected=o.track_id===selectedTrackId,r=layers.volumes&&occupancyFor(o)?(selected?5:2.5):(selected?9:6);
      ctx.fillStyle=selected?THEME.selected:conflict?THEME.objectMuted:THEME.object;ctx.beginPath();ctx.arc(s.x,s.y,r,0,Math.PI*2);ctx.fill();
      ctx.strokeStyle=selected?THEME.selected:conflict?THEME.objectMuted:THEME.object;ctx.lineWidth=1;
      if(conflict)ctx.setLineDash([2,3]);
      ctx.beginPath();ctx.arc(s.x,s.y,r+4,0,Math.PI*2);ctx.stroke();
      ctx.setLineDash([]);
      hitTargets.push({trackId:o.track_id,x:s.x,y:s.y});
      if(selected||positions.length<=12){
        labels.push({o:snapshot,s,selected,conflict});
      }
    }
    drawLabels(labels,w,h);
  }
  drawPair(basis,w,h);if(poseValid(poseIndex()))drawFrustum(pose,basis,w,h);drawAxes(basis,w,h);drawPlan(pose.p);drawVideoOverlay();
  $('sceneEmpty').hidden=visibleMapObjects().length>0;
  $('sceneEmpty').textContent=currentOnly?'当前帧暂无可用三维定位。仍可在视频中查看识别框。':
    '本段视频暂无物体三维定位证据。场景点和轨迹可查看，物体仍可在视频中识别。';
}
function drawPlan(cameraPosition){
  const el=$('plan'),g=resizeCanvas(el),w=el.clientWidth,h=el.clientHeight;
  g.fillStyle=THEME.plan;g.fillRect(0,0,w,h);
  const box=planScene,scale=Math.min((w-30)/Math.max(box.max[0]-box.min[0],1),(h-30)/Math.max(box.max[2]-box.min[2],1));
  const p2=p=>[w/2+(p[0]-box.center[0])*scale,h/2-(p[2]-box.center[2])*scale];
  if(layers.trajectory){
    g.strokeStyle=THEME.trajectory;g.lineWidth=1.4;g.beginPath();traj.forEach((p,i)=>{if(!trajectoryVisible(i))return;const q=p2(p);if(trajectoryEdgeVisible(i))g.lineTo(...q);else g.moveTo(...q);});g.stroke();
  }
  if(layers.objects)for(const o of visibleMapObjects()){
    const volume=layers.volumes?occupancyFor(o):null;
    if(volume){
      const lo=p2(volume.min),hi=p2(volume.max);
      g.fillStyle=o.track_id===selectedTrackId?'rgba(185,104,76,.28)':'rgba(150,150,144,.48)';
      g.fillRect(Math.min(lo[0],hi[0]),Math.min(lo[1],hi[1]),Math.abs(hi[0]-lo[0]),Math.abs(hi[1]-lo[1]));
      g.strokeStyle=o.track_id===selectedTrackId?THEME.selected:'#85857e';g.lineWidth=1;g.setLineDash([3,2]);
      g.strokeRect(Math.min(lo[0],hi[0]),Math.min(lo[1],hi[1]),Math.abs(hi[0]-lo[0]),Math.abs(hi[1]-lo[1]));g.setLineDash([]);
    }
    g.fillStyle=o.track_id===selectedTrackId?THEME.selected:positionSnapshot(o).position_temporal_consistency==='contradicted'?THEME.objectMuted:THEME.object;
    g.beginPath();g.arc(...p2(currentPosition(o)),3,0,Math.PI*2);g.fill();
  }
  if(traj.length&&poseValid(poseIndex())){g.fillStyle=THEME.camera;g.beginPath();g.arc(...p2(cameraPosition),4,0,Math.PI*2);g.fill();}
  g.fillStyle=THEME.planText;g.font='10px Geist, "Microsoft YaHei", sans-serif';g.fillText('Z ↑',8,15);g.fillText('X →',w-35,h-9);
}
function drawVideoOverlay(){
  if(!hasVideo)return;
  const el=$('videoOverlay'),g=resizeCanvas(el),w=el.clientWidth,h=el.clientHeight;g.clearRect(0,0,w,h);
  if(!video.videoWidth||!video.videoHeight)return;
  const frame=sourceFrame();
  let boxes=observationFrames.get(frame)||[];
  // Old runs have only the last object box; never pretend it is a current box.
  if(!observations.length)boxes=objs.filter(o=>Math.abs(video.currentTime-timeForFrame(o.frame_id))<.025);
  const scale=Math.min(w/video.videoWidth,h/video.videoHeight),left=(w-video.videoWidth*scale)/2,top=(h-video.videoHeight*scale)/2;
  for(const o of boxes.filter(o=>candidateVisible(o)&&(!activeClass||displayedClass(o)===activeClass))){
    if(!o.xyxy)continue;const [x1,y1,x2,y2]=o.xyxy;
    const selected=o.track_id===selectedTrackId;g.strokeStyle=selected?THEME.selected:THEME.videoBox;g.lineWidth=selected?2.5:1.5;
    g.strokeRect(left+x1*scale,top+y1*scale,(x2-x1)*scale,(y2-y1)*scale);
    const label=objectName(o),x=left+x1*scale,y=Math.max(14,top+y1*scale-3);
    g.font='10px Geist, "Microsoft YaHei", sans-serif';g.fillStyle=THEME.videoLabel;g.fillRect(x,y-12,g.measureText(label).width+8,15);
    g.fillStyle=selected?THEME.selected:THEME.videoText;g.fillText(label,x+3,y);
  }
  llmUI.drawSelection(g,{scale,left,top});
}
function field(dl,label,value){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;dl.append(dt,dd);}
function renderDetails(){
  const selected=objs.find(o=>o.track_id===selectedTrackId);if(!selected)return;
  const o=positionSnapshot(selected);
  const el=$('objectDetails');el.replaceChildren();$('selectedTag').textContent=statusOf(o);
  const name=document.createElement('strong');name.textContent=objectName(o);el.append(name);
  const dl=document.createElement('dl');
  const manual=classOverride(o);
  if(manual){field(dl,'人工核对',manual.class_name+'（你已确认）');field(dl,'原模型类别',o.manual_llm_object?'原模型未检出此物体':classLabel(o.class_name));}
  if(selected.manual_llm_object){
    field(dl,'类别来源','三张局部图的大模型建议，经你确认');field(dl,'三维位置','尚无三维几何证据，仅在选定视频帧标注');
    field(dl,'已选时刻',(selected.manual_selections||[]).map(s=>(s.timestamp??timeForFrame(s.frame_id)).toFixed(2)+' s').join(' / '));
    el.append(dl);const actions=document.createElement('div');
    for(const selection of selected.manual_selections||[]){const button=document.createElement('button');button.type='button';button.textContent='回看 '+(selection.timestamp??timeForFrame(selection.frame_id)).toFixed(2)+' s';button.addEventListener('click',()=>seekFrame(selection.frame_id));actions.append(button);}
    el.append(actions);$('reviewClassBtn').disabled=true;return;
  }
  if(o.observation_current===false)field(dl,'当前观测',`源帧 ${sourceFrame()} 未观测到此物体；`+
    (o.frame_id==null?'此时刻之前没有识别统计。':`下列类别与分数为第 ${o.frame_id} 帧历史统计。`));
  if(needsClassReview(o)&&!manual)field(dl,'类别核对',o.classification_evidence?.source==='image_structure_rule'?
    '孔阵列只说明表面结构，物体类型尚未确认，请回看原图核对':'尚未确认，请回看原图核对模型猜测');
  if(o.classification_evidence?.source==='image_structure_rule')field(dl,'类别来源','图像孔阵列结构候选（独立规则，不是类别模型）');
  if(o.classification_evidence?.source==='open_vocabulary_model'){
    field(dl,'类别来源','开放词汇模型候选，请结合原画面核对');
    field(dl,'类别提示词',o.classification_evidence.raw_model_class||o.class_name);
  }
  const cropCheck=o.classification_evidence?.crop_verification||o.detection_evidence?.crop_verification;
  if(cropCheck){
    field(dl,'原检测候选',classLabel(cropCheck.proposal?.class_name||o.class_name));
    field(dl,'局部图像核对',(cropCheck.top_candidates||[]).map(row=>
      `${classLabel(row.class_name)} ${Number(row.cosine_similarity).toFixed(3)}`).join(' / ')+'（相似度，非概率）');
    field(dl,'类别状态','两个模型共同提供候选，物体边界和类别仍需核对');
  }
  field(dl,'世界坐标',finitePoint(o.xyz)?o.xyz.map(fmt).join(' / ')+' units':'暂无定位证据');
  const volume=occupancyFor(selected);
  if(volume){
    field(dl,'占用尺寸 · X/Y/Z',volume.size.map(fmt).join(' / ')+' units');
    field(dl,'推估占用体积',fmt(volume.proxy_volume_map_units3??volume.volume_map_units3)+' units³');
    field(dl,'体积依据',volume.multi_view?.status==='constrained'?
      `${volume.multi_view.view_count} 个视角共同约束，未知厚度仍为推估`:'单视角检测框与深度，未知厚度按比例推估');
    if(volume.volume_sensitivity)field(dl,'厚度假设影响',
      fmt(volume.volume_sensitivity.min)+'–'+fmt(volume.volume_sensitivity.max)+' units³（非测量误差区间）');
    if(Number.isFinite(volume.aabb_volume_map_units3))field(dl,'外包围盒体积',
      fmt(volume.aabb_volume_map_units3)+' units³（用于候选间隙比较）');
    field(dl,'碰撞用途',volume.candidate_overlap_usable&&volume.frame_id===sourceFrame()?
      '可比较候选包围体，完整边界尚未核实':'仅空间预览，当前碰撞证据不足');
  }else field(dl,'占用体积','当前体积未知，保留位置点');
  const structureClass=o.classification_evidence?.source==='image_structure_rule';
  field(dl,cropCheck?'原检测候选分数':structureClass?'孔阵列匹配分数':'模型分数',
    Number.isFinite(o.confidence)?`${Math.round(o.confidence*100)}%`+(cropCheck?'（原框分数，非核对类别概率）':'（不是准确率）'):'此时刻无记录');
  field(dl,'类别一致性',Number.isFinite(o.classification_consistency)?`${Math.round(o.classification_consistency*100)}% / 近 ${o.classification_observations||1} 次`:'旧数据未记录');
  field(dl,'三维依据',SOURCE[o.position_source||o.depth_source]||'旧数据未核验');
  const evidence=o.evidence||{};
  field(dl,'定位观测',`${o.position_observations||0} 次 / ${evidence.sample_count??'—'} 个支撑点`);
  field(dl,'位置散布',Number.isFinite(o.position_uncertainty)?`${fmt(o.position_uncertainty)} units（不含相机误差）`:'未记录');
  field(dl,'画面一致性',o.position_temporal_consistency==='contradicted'?'与当前检测框矛盾，暂停关系判断':
    o.position_temporal_consistency==='consistent'?'投影落在检测框内（仍非精度证明）':'尚未复核');
  if(o.geometry)field(dl,'空间关系',o.position_relation_usable?'当前同帧多视证据可用':'本帧证据不足，保留画面方位');
  field(dl,'定位帧',o.position_frame==null?'未记录':String(o.position_frame));
  field(dl,o.observation_current===false?'历史识别统计':'末次识别',o.frame_id==null?'此时刻之前没有记录':`第 ${o.frame_id} 帧 / ${o.observations??0} 次观测`);
  const classification=o.classification_evidence||{};
  if(classification.latest_raw_class&&classification.latest_raw_class!==o.class_name)field(dl,'本帧原始类',classLabel(classification.latest_raw_class)+'（多帧投票保持显示类）');
  if(classification.alternatives?.length)field(dl,'其他候选',classification.alternatives.map(a=>classLabel(a.class_name)).join(' / '));
  if(['laptop','tv'].includes(o.class_name))field(dl,'类别范围','模型没有平板类别，屏幕设备需人工核对');
  el.append(dl);
  const buttons=document.createElement('div');buttons.style.marginTop='12px';
  for(const [label,frame] of [['回看识别帧',o.frame_id],['回看定位帧',o.position_frame]]){
    if(frame==null)continue;const button=document.createElement('button');button.textContent=label;
    button.addEventListener('click',()=>{if(label==='回看定位帧'){currentOnly=true;$('timeScope').checked=true;}seekFrame(frame);});buttons.append(button);
  }
  el.append(buttons);
  $('reviewClassBtn').disabled=false;
}
function seekFrame(frame){
  setPlaying(false);tNorm=frameIds.length>1?indexForFrame(frame)/(frameIds.length-1):0;
  if(hasVideo)video.currentTime=timeForFrame(frame);dirty=true;
}
function selectObject(o,seek=true){
  if(!o)return;
  // Selection never changes the video time, orbit pivot or zoom.
  selectedTrackId=o.track_id;
  renderObjectList();renderDetails();dirty=true;
  notifyMotion('object',{label:objectName(positionSnapshot(o)),trackId:o.track_id});
}
function focusSelection(){
  const selected=objs.find(o=>o.track_id===selectedTrackId),point=selected?currentPosition(selected):null;
  if(multipleWorlds&&selected&&!visibleMapObjects().includes(selected)){
    notifyMotion('toast',{label:'此记录属于其他空间段，请先回看对应视频时刻'});return;
  }
  if(finitePoint(point)){
    const snapshot=positionSnapshot(selected),radius=Math.max((snapshot.position_uncertainty||0)*1.6,scene.radius*.035,.04);
    const corners=occupancyCorners(selected);
    if(corners.length){fit(corners);return;}
    rememberCamera('focus');cancelCameraDrag();leaveCameraMode();freeLens=null;
    moveCamera({target:point.slice(),dist:Math.max(radius*4,.12),lensMix:0,zoomInput:0},true);
    notifyMotion('view',{label:'聚焦 '+objectName(snapshot)});return;
  }
  const points=visibleMapObjects().map(currentPosition);fit(points.length?points:fittedScenePoints());
}
function renderObjectList(){
  const list=$('objectList');list.replaceChildren();
  const all=allObjects().sort((a,b)=>Number(mapObjs.includes(b))-Number(mapObjs.includes(a))||b.observations-a.observations);
  $('objectCount').textContent=`${all.length} 条跟踪记录`;
  if(!all.length){const p=document.createElement('p');p.className='empty';p.textContent='暂无此类别的观测';list.append(p);}
  for(const o of all){
    const row=document.createElement('button');row.type='button';
    row.className='object-row'+(mapObjs.includes(o)?'':' estimate')+(o.track_id===selectedTrackId?' active':'');
    const snapshot=positionSnapshot(o);
    const label=document.createElement('strong');label.textContent=objectName(snapshot);
    const tag=document.createElement('span');tag.className='tag';tag.textContent=classOverride(o)?'你已确认':needsClassReview(snapshot)?'类别待核对':snapshot.position_temporal_consistency==='contradicted'?'待复核':finitePoint(snapshot.xyz)?'3D 候选':'2D';
    const detail=document.createElement('small');detail.textContent=`${snapshot.observations} 次观测 · ${snapshot.observation_current===false?'历史统计 · ':''}${statusOf(snapshot)}`;
    row.append(label,tag,detail);row.addEventListener('click',()=>selectObject(o));list.append(row);
  }
}
function updateRelation(){
  updateOccupancyReadout();
  const el=$('relationReadout');el.replaceChildren();const [objectA,objectB]=pairObjects();
  const basis=$('relationBasis').value==='world'?'world':'current';
  const relation=relationState(objectA,objectB,basis);
  if(relation.screen){const screen=document.createElement('div');screen.textContent=`当前画面：B 在 A ${relation.screen}。`;el.append(screen);}
  if(!relation.usable){const note=document.createElement('p');note.className='micro';note.textContent=relation.reason;el.append(note);return;}
  const {a,b,delta:d,distance,uncertainty}=relation;
  const metrics=document.createElement('div');metrics.className='metrics';
  for(const [label,value] of [['间距估计',fmt(distance)],['ΔX · 右',signed(d[0])],['ΔY · 下',signed(d[1])],['ΔZ · 前',signed(d[2])]]){
    const item=document.createElement('div');item.className='metric';const small=document.createElement('small'),big=document.createElement('b');
    small.textContent=label;big.textContent=value;item.append(small,big);metrics.append(item);
  }
  el.append(metrics);
  const text=document.createElement('div');
  const phrases=[Math.abs(d[0])<=uncertainty?'左右排序待核对':d[0]>0?'右侧':'左侧',
    Math.abs(d[1])<=uncertainty?'上下排序待核对':d[1]>0?'下方':'上方',
    Math.abs(d[2])<=uncertainty?'前后排序待核对':d[2]>0?'前方':'后方'];
  text.textContent=`空间中 B 相对 A：${phrases.join(' · ')}。方向沿${basis==='world'?'首帧世界轴':'当前镜头轴'}。`;el.append(text);
  const note=document.createElement('p');note.className='micro';
  note.textContent=`定位来自第 ${a.position_frame??a.frame_id} / ${b.position_frame??b.frame_id} 帧；`+
    '画面方位来自同帧检测框；空间分量来自多视几何。单位为相对地图单位，散布不含相机误差。';el.append(note);
  if(a.geometry?.position_relation_kind==='surface_support'||b.geometry?.position_relation_kind==='surface_support'){
    const surface=document.createElement('p');surface.className='micro';surface.textContent='此处比较持续观测到的物体表面位置，物体中心偏差尚未知。';el.append(surface);
  }
}
function informativeFrame(){
  let bestFrame=null,bestScore=-1;
  for(const [frame,rows] of observationFrames){
    if(DATA.summary?.frame_quality?.[String(frame)]?.pose_valid!==true)continue;
    const available=rows.filter(o=>finitePoint(o.geometry?.xyz));
    if(available.length<2)continue;
    const accepted=available.filter(o=>o.geometry.position_relation_usable);
    const score=accepted.length*10+available.length;
    if(score>bestScore){bestScore=score;bestFrame=frame;}
  }
  return bestFrame;
}
function showInformativeFrame(){
  const frame=informativeFrame();if(frame===null)return;
  currentOnly=true;$('timeScope').checked=true;seekFrame(frame);
  const rows=observationFrames.get(frame)||[],accepted=rows.filter(o=>o.geometry?.position_relation_usable);
  const pair=accepted.length>=2?accepted:rows.filter(o=>finitePoint(o.geometry?.xyz));
  if(pair.length>=2){$('objectA').value=String(pair[0].track_id);$('objectB').value=String(pair[1].track_id);}
  const points=rows.flatMap(o=>[...(finitePoint(o.geometry?.xyz)?[o.geometry.xyz]:[]),
    ...(o.geometry?.occupancy?.available?occupancyRenderer?.corners(o.geometry.occupancy)||[]:[])]);
  if(points.length)fit(points);
  renderObjectList();dirty=true;
}
function relationState(objectA,objectB,basis='current'){
  if(!objectA||!objectB||objectA===objectB)return {usable:false,reason:'请选择两个不同的目标。'};
  const a=positionSnapshot(objectA,hasGeometry),b=positionSnapshot(objectB,hasGeometry);
  const current=(observationFrames.get(sourceFrame())||[]);
  const oa=current.find(o=>o.track_id===objectA.track_id),ob=current.find(o=>o.track_id===objectB.track_id);
  let screen=null;
  if(oa?.xyxy&&ob?.xyxy){
    const dx=(ob.xyxy[0]+ob.xyxy[2]-oa.xyxy[0]-oa.xyxy[2])/2;
    const dy=(ob.xyxy[1]+ob.xyxy[3]-oa.xyxy[1]-oa.xyxy[3])/2;
    screen=[Math.abs(dx)<3?'左右接近':dx>0?'右侧':'左侧',Math.abs(dy)<3?'高度接近':dy>0?'下方':'上方'].join(' · ');
  }
  const unavailable=reason=>({usable:false,reason,a,b,screen});
  if(!finitePoint(a.xyz)||!finitePoint(b.xyz))return unavailable('当前帧没有两处同时可用的三维位置，保留画面方位。');
  if([a,b].some(o=>o.position_temporal_consistency==='contradicted'))return unavailable('位置与当前画面矛盾，暂不显示空间距离及深度排序。');
  if(hasGeometry){
    if(!a.geometry||!b.geometry||a.geometry.frame_id!==sourceFrame()||b.geometry.frame_id!==sourceFrame())return unavailable('这两个位置不是当前同一帧的观测，暂不组合空间关系。');
    if(!a.world_frame_id||a.world_frame_id!==b.world_frame_id)return unavailable('两个目标不在同一个重建坐标系中，不能比较。');
    if(DATA.summary?.scale_consistent!==true)return unavailable('本段尚未建立统一的重建尺度。');
    const quality=DATA.summary?.frame_quality?.[String(sourceFrame())];
    if(quality?.pose_valid!==true||a.geometry.pose_valid!==true||b.geometry.pose_valid!==true)return unavailable('当前相机定位不可靠，只保留画面方位。');
    const poseIndex=indexForFrame(sourceFrame());
    const cameraWorld=DATA.trajectory_world_frame_ids?.[poseIndex]||quality.world_frame_id||DATA.summary.world_frame_id;
    if(!cameraWorld||cameraWorld!==a.world_frame_id)return unavailable('当前相机与两个目标不属于同一个重建坐标系。');
    if(a.position_relation_usable!==true||b.position_relation_usable!==true)return unavailable('多视位置证据尚不足，暂不判断远近或空间距离。');
  }
  let d=sub(b.xyz,a.xyz);
  if(basis==='current'){
    const i=indexForFrame(sourceFrame()),R=rotations[i];
    if(!R||frameIds[i]!==sourceFrame())return unavailable('没有当前帧的镜头朝向，无法按镜头解释空间方向。');
    d=[0,1,2].map(k=>R.reduce((sum,row,j)=>sum+row[k]*d[j],0));
  }
  return {usable:true,a,b,screen,delta:d,distance:Math.hypot(...d),
    uncertainty:(a.position_uncertainty||0)+(b.position_uncertainty||0)};
}
function populatePairs(){
  const comparableObjects=objs.filter(o=>!isSupplementCandidate(o)&&!o.manual_llm_object);
  for(const id of ['objectA','objectB']){
    const select=$(id),previous=select.value;select.replaceChildren();select.options.length=0;
    for(const o of comparableObjects)select.add(new Option(objectName(positionSnapshot(o)),String(o.track_id)));
    select.disabled=comparableObjects.length<2;
    if(comparableObjects.some(o=>String(o.track_id)===previous))select.value=previous;
    else if(id==='objectB'&&comparableObjects.length>1)select.selectedIndex=1;
    if(!select.visualMapRelationBound){select.addEventListener('change',()=>{updateRelation();dirty=true;});select.visualMapRelationBound=true;}
  }
  $('pairBtn').disabled=comparableObjects.length<2;updateRelation();
}
function refreshObjectSelectorNames(){
  const byTrack=new Map(objs.map(o=>[String(o.track_id),o]));
  for(const id of ['objectA','objectB','llmTarget'])for(const option of $(id).options){
    const o=byTrack.get(option.value);if(o)option.text=objectName(positionSnapshot(o));
  }
}
function setPlaying(on){
  playing=on;$('playBtn').textContent=on?'⏸ 暂停':'▶ 播放';
  if(hasVideo){if(on)video.play().catch(()=>setPlaying(false));else video.pause();}
  dirty=true;
  notifyMotion('playback',{label:on?'播放视频':'暂停视频',paused:!on});
}
function updateHud(){
  updateZoomControls();
  const frame=currentFrame(),pose=poseAt();
  const valid=poseValid(poseIndex()),world=currentWorld(),segment=submaps.findIndex(row=>row.world_frame_id===world)+1;
  $('poseHud').textContent=valid?`相机处理帧 ${frame}${multipleWorlds?` · 空间段 ${segment}`:''} · ${pose.p.map(v=>v.toFixed(1)).join(' / ')} units`:
    `源帧 ${sourceFrame()} · 三维定位暂时丢失`;
  $('trackingNotice').setAttribute('data-state',valid?(multipleWorlds?'submap':'tracked'):'missing');
  $('trackingNotice').textContent=!valid?'此时刻相机定位丢失，暂停三维镜头更新；视频与识别继续。恢复后的独立空间段会自动显示。':
    multipleWorlds?`正在显示空间段 ${segment} / ${submaps.length}。各段独立重建，尚未对齐坐标与尺度，不能跨段比较位置或体积。`:
    '当前帧相机定位有效。时间轴下方绿色为有效定位，棕色为暂缺定位。';
  $('timeLabel').textContent=`${normToTime(tNorm).toFixed(1)} s`;
  $('videoTime').textContent=`源帧 ${sourceFrame()} · ${(hasVideo?video.currentTime:normToTime(tNorm)).toFixed(1)} s`;
  $('seek').value=String(Math.round(tNorm*1000));
}
function syncSubmapView(){
  const world=currentWorld();if(!multipleWorlds||world===displayedWorld)return;
  displayedWorld=world;cameraMotion=null;cameraHistory.length=0;cancelCameraDrag();
  scene=bounds([...activeTrajectory(),...robustCloud(activePoints())]);planScene=scene;
  target=scene.center.slice();dist=scene.radius*2.8;zoomInput=0;freeLens=null;lensMix=0;
  projectionMix=1;cameraUp=[0,-1,0];pendingMode=null;follow=false;
  if(mode!=='camera'){mode='perspective';yaw=.62;pitch=.48;}
  syncCameraControls();dirty=true;
}
function renderTrackingCoverage(){
  const segments=DATA.summary?.geometry_coverage?.segments||[];
  $('trackingCoverage').replaceChildren();
  for(const segment of segments){
    const button=document.createElement('button');button.className='tracking-segment';button.style.flex=String(segment.frames);
    button.setAttribute('data-state',!segment.pose_valid?'missing':multipleWorlds?'submap':'tracked');
    const label=`${segment.start_seconds.toFixed(1)}–${segment.end_seconds.toFixed(1)} 秒 · ${segment.pose_valid?'相机定位有效':'定位暂缺'}`;
    button.title=label;button.setAttribute('aria-label',label);
    button.addEventListener('click',()=>{tNorm=timeToNorm(segment.start_seconds);if(hasVideo)video.currentTime=segment.start_seconds;syncSubmapView();dirty=true;});
    $('trackingCoverage').append(button);
  }
}
function localPointer(e){const rect=canvas.getBoundingClientRect();return {x:e.clientX-rect.left,y:e.clientY-rect.top};}
function pickAt(x,y){
  // Markers are drawn above the faces, so an explicit marker click wins.
  return hitTargets.slice().reverse().find(p=>Math.hypot(p.x-x,p.y-y)<18)||occupancyRenderer?.pickAt(x,y,volumeHitTargets);
}
function worldAtScreen(x,y,basis=viewerBasis(),planePoint=target){
  // For the free camera the focus plane has a constant depth. Solve analytically:
  // iterative inversion of a strong video-lens distortion has several roots.
  if(mode!=='camera'&&(!freeLens||lensMix<1e-5)){
    const depth=dot(sub(planePoint,basis.eye),basis.z);
    if(depth<=0||!Number.isFinite(depth))return planePoint.slice();
    const scale=Math.min(canvas.clientWidth,canvas.clientHeight)*(1.15/depth*projectionMix+1.5/dist*(1-projectionMix))*zoomFactor();
    const projected=project(planePoint,basis,canvas.clientWidth,canvas.clientHeight);
    if(!projected||!Number.isFinite(scale)||scale<1e-8)return planePoint.slice();
    return add(planePoint,add(mul(basis.x,(x-projected.x)/scale),mul(basis.y,(projected.y-y)/scale)));
  }
  let point=planePoint.slice();
  const step=Math.max(dist*.001,.00001);
  for(let i=0;i<7;i++){
    const p=project(point,basis,canvas.clientWidth,canvas.clientHeight);
    const px=project(add(point,mul(basis.x,step)),basis,canvas.clientWidth,canvas.clientHeight);
    const py=project(add(point,mul(basis.y,step)),basis,canvas.clientWidth,canvas.clientHeight);
    if(!p||!px||!py)break;
    const a=(px.x-p.x)/step,b=(py.x-p.x)/step,c=(px.y-p.y)/step,d=(py.y-p.y)/step,det=a*d-b*c;
    if(Math.abs(det)<1e-10)break;
    const ex=x-p.x,ey=y-p.y;
    point=add(point,add(mul(basis.x,(d*ex-b*ey)/det),mul(basis.y,(-c*ex+a*ey)/det)));
    if(Math.hypot(ex,ey)<.001)break;
  }
  return point;
}
function zoomAt(x,y,wheelDelta){
  if(![x,y,wheelDelta].every(Number.isFinite))return;
  rememberCamera('wheel');
  if(cameraMotion||pendingMode)leaveCameraMode();
  // Cursor movement must not smuggle a pan into a zoom gesture. A smooth bounded
  // mapping avoids hard saturation losing the reverse wheel's accumulated amount.
  zoomInput-=wheelDelta*.0012;
  dirty=true;updateZoomControls();
  motion?.drag('zoom','move',clamp(.5+.5*Math.log(zoomFactor())/zoomExtent,0,1));
}
function setZoomFactor(factor){
  if(!Number.isFinite(factor)||factor<=0)return;
  rememberCamera('zoom');if(cameraMotion||pendingMode)leaveCameraMode();
  zoomInput=Math.sinh(zoomExtent*Math.atanh(clamp(Math.log(factor)/zoomExtent,-.99999,.99999)));
  dirty=true;updateZoomControls();
}
function nudgeView(horizontal,vertical){
  const baseYaw=cameraMotion?.kind==='nudge'?cameraMotion.goal[4]:yaw;
  const basePitch=cameraMotion?.kind==='nudge'?cameraMotion.goal[5]:pitch;
  rememberCamera('rotate');cancelCameraDrag();leaveCameraMode();normalizeNavigationLens();
  mode=projectionMix===0?'orthographic':'perspective';
  moveCamera({yaw:baseYaw+horizontal*Math.PI/12,pitch:clamp(basePitch+vertical*Math.PI/18,-Math.PI/2+.0001,Math.PI/2-.0001),up:[0,-1,0]},true);
  if(cameraMotion)cameraMotion.kind='nudge';
  syncCameraControls();
}
let drag=null;
function cancelCameraDrag(){
  if(!drag)return;
  const previous=drag;drag=null;motion?.drag(previous.kind,'cancel',0);
  if(canvas.hasPointerCapture?.(previous.pointerId))canvas.releasePointerCapture(previous.pointerId);
}
canvas.addEventListener('pointerdown',e=>{
  if(drag)return;
  e.preventDefault();canvas.focus?.({preventScroll:true});
  const pan=e.button===1||e.button===2||e.shiftKey;
  drag={pointerId:e.pointerId,button:e.button,kind:pan?'pan':'rotate',x:e.clientX,y:e.clientY,
    startX:e.clientX,startY:e.clientY,moved:false};
  canvas.setPointerCapture(e.pointerId);
  motion?.drag(drag.kind,'start',0);
});
canvas.addEventListener('pointermove',e=>{
  if(!drag||e.pointerId!==drag.pointerId)return;
  const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
  if(!drag.moved&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<=4)return;
  if(!drag.moved){rememberCamera('drag');leaveCameraMode();normalizeNavigationLens();drag.moved=true;}
  if(drag.kind==='pan'){
    const rect=canvas.getBoundingClientRect(),basis=viewerBasis();
    const from=worldAtScreen(drag.x-rect.left,drag.y-rect.top,basis);
    const to=worldAtScreen(e.clientX-rect.left,e.clientY-rect.top,basis);
    target=boundedTarget(add(target,sub(from,to)));
  }else{
    const speed=Math.PI/Math.max(canvas.clientHeight,400);
    yaw=Math.atan2(Math.sin(yaw+dx*speed),Math.cos(yaw+dx*speed));
    pitch=clamp(pitch+dy*speed,-Math.PI/2+.0001,Math.PI/2-.0001);
    cameraUp=[0,-1,0];mode=projectionMix===0?'orthographic':'perspective';
    syncCameraControls();
  }
  drag.x=e.clientX;drag.y=e.clientY;dirty=true;
  motion?.drag(drag.kind,'move',Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)/Math.max(canvas.clientWidth,1));
});
function releaseDrag(e,cancel=false){
  if(!drag||e.pointerId!==drag.pointerId)return;
  if(!cancel&&!drag.moved&&drag.button===0){
    const p=localPointer(e),hit=pickAt(p.x,p.y);
    if(hit)selectObject(objs.find(o=>o.track_id===hit.trackId),false);
  }
  motion?.drag(drag.kind,'end',0);drag=null;
  if(canvas.hasPointerCapture?.(e.pointerId))canvas.releasePointerCapture(e.pointerId);
}
canvas.addEventListener('pointerup',e=>releaseDrag(e));
canvas.addEventListener('pointercancel',e=>releaseDrag(e,true));
canvas.addEventListener('lostpointercapture',e=>releaseDrag(e,true));
canvas.addEventListener('dblclick',e=>{
  e.preventDefault();const p=localPointer(e),hit=pickAt(p.x,p.y);
  if(hit)selectObject(objs.find(o=>o.track_id===hit.trackId),false);
  focusSelection();
});
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('wheel',e=>{
  e.preventDefault();const p=localPointer(e),delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?canvas.clientHeight:1);
  zoomAt(p.x,p.y,delta);
},{passive:false});
$('zoomOut').addEventListener('click',()=>zoomAt(0,0,160));
$('zoomIn').addEventListener('click',()=>zoomAt(0,0,-160));
$('zoomReset').addEventListener('click',()=>setZoomFactor(1));
$('zoomRange').addEventListener('input',e=>{
  rememberCamera('zoom');if(cameraMotion||pendingMode)leaveCameraMode();
  zoomInput=(clamp(Number(e.target.value),0,1000)-500)/25;dirty=true;updateZoomControls();
});
for(const [id,h,v] of [['rotateLeft',-1,0],['rotateRight',1,0],['rotateUp',0,-1],['rotateDown',0,1]])
  $(id).addEventListener('click',()=>nudgeView(h,v));
for(const id of ['perspective','top','front','side','camera'])$(id+'Btn').addEventListener('click',()=>setView(id));
$('fitBtn').addEventListener('click',()=>fit(fittedScenePoints(),true));
$('objectsBtn').addEventListener('click',()=>{const p=[...visibleMapObjects().map(currentPosition),...visibleOccupancies().flatMap(o=>occupancyRenderer.corners(o.bounds))];if(p.length)fit(p);});
$('focusBtn').addEventListener('click',focusSelection);
$('pairBtn').addEventListener('click',()=>{const [a,b]=pairObjects();if(a&&b){
  if(multipleWorlds&&(!visibleMapObjects().includes(a)||!visibleMapObjects().includes(b))){notifyMotion('toast',{label:'两个物体需在当前同一空间段中'});return;}
  const p=[currentPosition(a),currentPosition(b)];if(p.every(finitePoint))fit([...p,...occupancyCorners(a),...occupancyCorners(b)]);}});
$('followBtn').addEventListener('click',()=>{rememberCamera('follow');const nextFollow=!follow;
  cancelCameraDrag();leaveCameraMode();normalizeNavigationLens();follow=nextFollow;
  syncCameraControls();dirty=true;notifyMotion('view',{label:follow?'跟随镜头位置':'自由查看'});});
$('resetBtn').addEventListener('click',()=>fit(fittedScenePoints(),true));
$('resetBtn').setAttribute('title','恢复全景相机，保留当前视频时间和选择 · R / Home');
$('playBtn').addEventListener('click',()=>{if(tNorm>=1)seekFrame(frameIds[0]||0);setPlaying(!playing);});
$('seek').addEventListener('input',()=>{setPlaying(false);tNorm=Number($('seek').value)/1000;if(hasVideo)video.currentTime=normToTime(tNorm);dirty=true;notifyMotion('scrub',{label:normToTime(tNorm).toFixed(1)+' s'});});
for(const [id,key] of [['layerTrajectory','trajectory'],['layerPoints','points'],['layerObjects','objects'],['layerVolumes','volumes'],['layerGrid','grid'],['layerUncertainty','uncertainty']]){
  $(id).addEventListener('change',e=>{layers[key]=e.target.checked;dirty=true;notifyMotion('display',{label:'图层已更新'});});
}
$('occupancyMargin').addEventListener('input',()=>{updateOccupancyReadout();dirty=true;});
  $('timeScope').addEventListener('change',e=>{currentOnly=e.target.checked;renderObjectList();refreshObjectSelectorNames();dirty=true;});
$('relationBasis').addEventListener('change',()=>{updateRelation();dirty=true;});
$('evidenceFrameBtn').addEventListener('click',showInformativeFrame);
window.addEventListener('resize',()=>{dirty=true;});
function populateClassFilter(){
  const select=$('classFilter');select.replaceChildren();
  const counts=new Map();for(const o of objs){const name=displayedClass(o);counts.set(name,(counts.get(name)||0)+1);}
  select.add(new Option(`全部类别 (${objs.length})`,''));
  for(const [name,count] of [...counts].sort((a,b)=>b[1]-a[1]))select.add(new Option(`${classLabel(name)} (${count})`,name));
  select.value=activeClass;
}
populateClassFilter();
$('classFilter').addEventListener('change',()=>{activeClass=$('classFilter').value;
  if(activeClass&&objs.some(o=>o.class_name===activeClass&&isSupplementCandidate(o))){showCandidates=true;$('showCandidates').checked=true;}
  renderObjectList();dirty=true;notifyMotion('selection',{label:activeClass?classLabel(activeClass):'全部类别'});});
$('candidateCount').textContent=`(${objs.filter(isSupplementCandidate).length})`;
$('showCandidates').addEventListener('change',()=>{showCandidates=$('showCandidates').checked;
  if(!showCandidates&&objs.some(o=>o.track_id===selectedTrackId&&isSupplementCandidate(o))){selectedTrackId=null;
    $('selectedTag').textContent='选择一个物体';$('objectDetails').textContent='点选场景标记或下方清单，查看坐标、类别一致性与定位依据。';}
  renderObjectList();dirty=true;});
function openDialog(id){
  const el=$(id);if(typeof el.showModal==='function'){if(!el.open)el.showModal();}else el.hidden=false;
}
function closeDialog(id){const el=$(id);if(typeof el.close==='function')el.close();else el.hidden=true;}
const commands=[
  {label:'立体视角',key:'1',run:()=>setView('perspective')},
  {label:'顶视 · X / Z',key:'2',run:()=>setView('top')},
  {label:'正视 · X / Y',key:'3',run:()=>setView('front')},
  {label:'侧视 · Z / Y',key:'4',run:()=>setView('side')},
  {label:'沿视频镜头查看',key:'5',run:()=>setView('camera')},
  {label:'聚焦选中物体',key:'F',run:focusSelection},
  {label:'查看全景',key:'',run:()=>fit(fittedScenePoints(),true)},
  {label:'恢复全景相机',key:'R',run:()=>fit(fittedScenePoints(),true)},
  {label:'返回上一个视角',key:'Ctrl Z',run:undoCamera},
  {label:'聚焦当前物体',key:'',run:()=>{const p=visibleMapObjects().map(currentPosition);if(p.length)fit(p);}},
  {label:'播放 / 暂停视频',key:'Space',run:()=>setPlaying(!playing)},
  {label:'跳到空间证据较完整的画面',key:'',run:showInformativeFrame},
  {label:'显示 / 隐藏场景点',key:'',run:()=>{layers.points=!layers.points;$('layerPoints').checked=layers.points;dirty=true;notifyMotion('display',{label:'场景点 '+(layers.points?'显示':'隐藏')});}},
  {label:'显示 / 隐藏参考网格',key:'',run:()=>{layers.grid=!layers.grid;$('layerGrid').checked=layers.grid;dirty=true;notifyMotion('display',{label:'参考网格 '+(layers.grid?'显示':'隐藏')});}},
];
function renderCommands(){
  const list=$('commandResults'),query=$('commandSearch').value.trim().toLowerCase();list.replaceChildren();
  const matched=commands.filter(command=>command.label.toLowerCase().includes(query)||command.key.toLowerCase()===query);
  for(const command of matched){
    const button=document.createElement('button');button.type='button';
    const label=document.createElement('span'),key=document.createElement('kbd');
    label.textContent=command.label;key.textContent=command.key;button.append(label,key);
    button.addEventListener('click',()=>{closeDialog('commandDialog');command.run();});list.append(button);
  }
  if(!matched.length){const p=document.createElement('p');p.className='empty';p.textContent='没有匹配的操作';list.append(p);}
}
function showCommands(){
  $('commandSearch').value='';renderCommands();openDialog('commandDialog');$('commandSearch').focus?.();
  notifyMotion('command',{label:'快捷操作'});
}
$('commandBtn').addEventListener('click',showCommands);
$('commandSearch').addEventListener('input',renderCommands);
$('commandSearch').addEventListener('keydown',e=>{
  if(e.key==='Enter'){const query=$('commandSearch').value.trim().toLowerCase();
    const command=commands.find(c=>c.label.toLowerCase().includes(query)||c.key.toLowerCase()===query);
    if(command){e.preventDefault();closeDialog('commandDialog');command.run();}}
});
$('commandClose').addEventListener('click',()=>closeDialog('commandDialog'));
$('newVideoBtn').addEventListener('click',()=>notifyMotion('selection',{label:'选择新视频'}));
$('reviewClassBtn').addEventListener('click',()=>{
  const o=objs.find(o=>o.track_id===selectedTrackId);if(!o)return;
  $('classEdit').value=classLabel(displayedClass(positionSnapshot(o)))||'';
  openDialog('classDialog');$('classEdit').focus?.();notifyMotion('object',{label:'人工核对类别'});
});
$('classCancel').addEventListener('click',()=>closeDialog('classDialog'));
$('classSave').addEventListener('click',()=>{
  const o=objs.find(o=>o.track_id===selectedTrackId),name=$('classEdit').value.trim().slice(0,40);if(!o||!name)return;
  classOverrides.set(String(o.track_id),{track_id:o.track_id,class_name:name,
    model_class:o.class_name,source:'user_manual_review',reviewed_at:new Date().toISOString(),
    supersedes_label_id:llmClassOverrides.get(String(o.track_id))?.label_id});
  const saved=saveClassOverrides();
  closeDialog('classDialog');activeClass='';populateClassFilter();refreshObjectSelectorNames();renderObjectList();renderDetails();dirty=true;
  notifyMotion('toast',{label:saved?'人工核对已保存':'已修改显示，请导出保存核对记录'});
});
$('exportCorrectionsBtn').addEventListener('click',()=>{
  const blob=new Blob([JSON.stringify({source:'user_manual_review',video:DATA.title||DATA.video_filename,
    corrections:[...classOverrides.values()]},null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='visualmap_human_review.json';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);notifyMotion('toast',{label:'人工核对记录已导出'});
});
window.addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();showCommands();return;}
  if(e.key==='Escape'){closeDialog('commandDialog');closeDialog('classDialog');closeDialog('llmSettingsDialog');llmUI.cancelCapture();motion?.finish();return;}
  if($('commandDialog').open||$('classDialog').open||$('llmSettingsDialog').open||llmUI.isExpanded()||e.target?.isContentEditable||['INPUT','TEXTAREA','SELECT'].includes(e.target?.tagName))return;
  const key=e.key.toLowerCase();
  if((e.ctrlKey||e.metaKey)&&key==='z'){e.preventDefault();undoCamera();}
  else if(key==='r'||key==='home'){e.preventDefault();fit(fittedScenePoints(),true);}
  else if(key==='f'){e.preventDefault();focusSelection();}
  else if(/^[1-5]$/.test(key)){e.preventDefault();setView(['perspective','top','front','side','camera'][Number(key)-1]);}
  else if(e.code==='Space'&&e.target===canvas){e.preventDefault();setPlaying(!playing);}
});
window.addEventListener('blur',cancelCameraDrag);
if(typeof window.matchMedia==='function')window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener?.('change',()=>{
  if(reducedMotion()&&cameraMotion){const finalMode=cameraMotion.finishMode;applyCameraValues(cameraMotion.goal);
    cameraMotion=null;if(finalMode){mode=finalMode;pendingMode=null;syncCameraControls();}dirty=true;}
});
if(hasVideo){
  $('videoPanel').hidden=false;video.src=DATA.video_filename;
  for(const event of ['timeupdate','seeked','loadedmetadata'])video.addEventListener(event,()=>{dirty=true;});
  if(hasGeometry)video.addEventListener('loadedmetadata',showInformativeFrame,{once:true});
  video.addEventListener('error',()=>{hasVideo=false;$('alert').hidden=false;$('alert').textContent='原视频暂时无法播放。请将 source_video 与地图页面放在同一目录；三维视图仍可使用。';dirty=true;});
}
const summary=DATA.summary||{},detector=summary.detector||'unknown';
$('title').textContent=DATA.title||'视频中的空间关系';document.title=$('title').textContent;
$('detectorStatus').textContent=detector==='demo'?'颜色演示检测':detector==='none'?'未启用物体识别':'视频识别 · 多帧核对';
const baselineRecords=objs.filter(o=>!isSupplementCandidate(o)),baselinePositioned=mapObjs.filter(o=>!isSupplementCandidate(o));
for(const [label,valueText] of [['相机帧',traj.length],['场景点',summary.num_map_points??pts.length],['基线记录',baselineRecords.length],['有 3D 证据',baselinePositioned.length],['扩展候选',objs.length-baselineRecords.length]]){
  const chip=document.createElement('div');chip.className='chip';chip.append(document.createTextNode(label));
  const value=document.createElement('b'),v=valueText;value.textContent = String(v);chip.append(value);$('chips').append(chip);
}
const quality=$('qualityNotice'),heading=document.createElement('strong');heading.textContent='定位证据与类别分开看';quality.append(heading);
quality.append(document.createTextNode(detector==='demo'?'本次为色块演示，不能代表真实识别效果。':
  '模型分数和多帧一致性都不是准确率。每条记录可能是同一物体在不同时间的轨迹段；COCO 模型没有平板类别。'));
if(objs.some(isSupplementCandidate)){const p=document.createElement('p');p.className='micro';
  p.textContent='扩展类别候选默认收起，可在识别记录中开启核对。这些候选可能误类，不参与空间关系计算。';quality.append(p);}
const contradicted=mapObjs.filter(o=>o.position_temporal_consistency==='contradicted').length;
if(contradicted){const p=document.createElement('p');p.className='micro';p.textContent=`${contradicted} 条历史位置已与后续画面矛盾，灰色标记仅供回看，暂停关系判断。`;quality.append(p);}
if(Number.isFinite(summary.track_rate)){
  const p=document.createElement('p');p.className='micro';p.textContent=`相机有效跟踪帧 ${(summary.track_rate*100).toFixed(1)}%。丢跟踪和未标定内参会影响空间关系。`;quality.append(p);
}
const coverage=summary.geometry_coverage;
if(coverage?.status==='partial'){
  const p=document.createElement('p');p.className='micro';p.textContent=`三维定位覆盖 ${coverage.valid_pose_frames} / ${coverage.total_frames} 帧；视频读取完整与三维重建完整分别记录。此比例是有效位姿覆盖，不是准确率。`;quality.append(p);
}
if(!traj.length){$('alert').hidden=false;$('alert').textContent='没有相机轨迹。请先处理一段视频。';}
function loop(timestamp){
  sampleCameraMotion(timestamp/1000);
  const dt=clockLast?Math.min((timestamp-clockLast)/1000,.1):0;clockLast=timestamp;
  if(hasVideo){
    const u=timeToNorm(video.currentTime);if(Math.abs(u-tNorm)>1e-7){tNorm=u;dirty=true;}
    playing=!video.paused&&!video.ended;$('playBtn').textContent=playing?'⏸ 暂停':'▶ 播放';
  }else if(playing){
    const duration=times.length>1?times[times.length-1]-times[0]:20;
    tNorm=Math.min(1,tNorm+dt/Math.max(.1,duration));if(tNorm>=1)setPlaying(false);dirty=true;
  }
  syncSubmapView();
  if(follow&&dirty&&poseValid(poseIndex()))target=poseAt().p.slice();
  if(dirty){draw();updateHud();updateRelation();
    if(listFrame!==sourceFrame()){if(currentOnly)renderObjectList();refreshObjectSelectorNames();listFrame=sourceFrame();}
    if(selectedTrackId!==null)renderDetails();dirty=false;}
  requestAnimationFrame(loop);
}
// Manual image review is an explicit local-service action. This module never
// contacts a provider or sends the full video from the browser.
const llmUI=(()=>{
  const overlay=$('videoOverlay');
  const inferredRun=typeof location!=='undefined'?location.pathname.match(/^\/results\/(.+)\/semantic_map\.html$/)?.[1]:null;
  const runId=DATA.run_id||(inferredRun?decodeURIComponent(inferredRun):null),draftKey='visualmap:llm-draft:'+(runId||correctionHash);
  let service=null,settings=null,selections=[],capture=false,drag=null,busy=false,settingsBusy=false,reviewId=null,reviewComplete=false,confirmed=false,pollEpoch=0,videoExpanded=false,sameObjectWarning=false;
  const previews=new Map();
  const element=(tag,text,cls)=>{const n=document.createElement(tag);if(text!=null)n.textContent=String(text);if(cls)n.className=cls;return n;};
  const textValue=value=>Array.isArray(value)?value.map(textValue).join('；'):value&&typeof value==='object'?Object.values(value).map(textValue).join('；'):String(value??'');
  function status(text){$('llmStatus').textContent=text;}
  async function request(path,body){
    if(typeof fetch!=='function'||!runId)throw new Error('请从本机视频工作台打开此结果，再使用大模型辅助。');
    if(body&&!service?.token)throw new Error('尚未获得本机服务授权，请重新打开结果页面。');
    const response=await fetch(path,{method:body?'POST':'GET',cache:'no-store',headers:body?{'Content-Type':'application/json','X-VisualMap-Token':service.token}:undefined,body:body?JSON.stringify(body):undefined});
    let data;try{data=await response.json();}catch{throw new Error('本机服务未返回有效数据，请从视频工作台重新打开结果。');}
    if(!response.ok)throw new Error(data.error||'请求未完成，请稍后重试。');return data;
  }
  function saveDraft(){
    try{window.localStorage?.setItem(draftKey,JSON.stringify({selections:selections.map(({frame_id,xyxy,timestamp})=>({frame_id,xyxy,timestamp})),target:$('llmTarget').value,question:$('llmQuestion').value,review_id:reviewId,confirmed}));}catch{}
  }
  function setCapture(on){capture=on;drag=null;overlay.classList.toggle('llm-selecting',on);$('llmCapture').textContent=on?'取消框选':'框选当前画面';
    $('llmExpandedCapture').textContent=on?'取消框选':'暂停并框选此刻';
    $('llmCaptureHint').textContent=on?'在上方视频画面按住并拖出矩形；框住同一物体的完整边界。':'每次只框选一个物体，尽量保留完整边界。';controls();dirty=true;}
  function expandVideo(on){videoExpanded=on;$('videoPanel').classList.toggle('video-expanded',on);$('llmExpandedTools').hidden=!on;$('llmVideoExpand').textContent=on?'收起画面 ↙':'放大选图 ↗';if(!on)setCapture(false);dirty=true;}
  function controls(){
    $('llmCapture').disabled=busy||selections.length>=3||!hasVideo||!video.videoWidth||video.seeking||video.readyState<2;
    $('llmExpandedCapture').disabled=$('llmCapture').disabled;$('llmExpandedCount').textContent='已选 '+selections.length+' / 3 张'+(capture?' · 在画面上拖出矩形':' · 使用视频进度条切换时刻');
    $('llmReset').disabled=busy||Boolean(reviewId&&!reviewComplete)||(!selections.length&&!reviewId);$('llmSubmit').disabled=busy||selections.length!==3||!settings?.has_api_key||!service?.token||Boolean(reviewId);
    $('llmTarget').disabled=busy||Boolean(reviewId);$('llmQuestion').disabled=busy||Boolean(reviewId);
    $('llmConfirm').disabled=busy||!reviewComplete||confirmed||!$('llmConfirmedName').value.trim()||(sameObjectWarning&&!$('llmSameObjectAck').checked);
  }
  function populateTargets(preferred){
    const target=$('llmTarget'),previous=preferred??target.value;target.replaceChildren();target.options.length=0;
    target.add(new Option('＋ 补充本地模型漏检的新物体',''));
    for(const o of objs.filter(o=>!o.manual_llm_object))target.add(new Option(objectName(positionSnapshot(o)),String(o.track_id)));
    target.value=objs.some(o=>String(o.track_id)===String(previous)&&!o.manual_llm_object)?String(previous):'';
  }
  function previewFrom(source,selection){
    if(!source.videoWidth||!source.videoHeight)return null;
    const [x1,y1,x2,y2]=selection.xyxy,w=x2-x1,h=y2-y1;if(w<=0||h<=0)return null;
    const thumb=document.createElement('canvas'),scale=Math.min(1,220/Math.max(w,h));thumb.width=Math.max(1,Math.round(w*scale));thumb.height=Math.max(1,Math.round(h*scale));
    try{thumb.getContext('2d').drawImage(source,x1,y1,w,h,0,0,thumb.width,thumb.height);return thumb.toDataURL('image/jpeg',.8);}catch{return null;}
  }
  function restorePreview(selection){
    if(previews.has(selection.frame_id)||typeof document.createElement('canvas').toDataURL!=='function')return;
    const clip=document.createElement('video');clip.muted=true;clip.preload='auto';clip.playsInline=true;let done=false;
    const cleanup=()=>{if(done)return;done=true;clip.pause();clip.removeAttribute('src');clip.load();};
    clip.addEventListener('loadeddata',()=>{if(Math.abs(clip.currentTime-selection.timestamp)>.001)clip.currentTime=selection.timestamp;else finish();},{once:true});
    const finish=()=>{if(done)return;const preview=previewFrom(clip,selection);if(preview)previews.set(selection.frame_id,preview);cleanup();renderEvidence();};
    clip.addEventListener('seeked',finish,{once:true});clip.addEventListener('error',cleanup,{once:true});clip.src=DATA.video_filename;clip.load();
    setTimeout(cleanup,15000);
  }
  function renderEvidence(){
    const slots=[];
    for(let i=0;i<3;i++){
      const selection=selections[i],slot=element('div',null,'llm-evidence-slot'+(selection?' filled':''));
      if(selection){
        const preview=previews.get(selection.frame_id),jump=element('button',null,'llm-preview');jump.type='button';jump.setAttribute('aria-label','回看选图 '+(i+1)+'，'+selection.timestamp.toFixed(2)+' 秒');
        if(preview){const image=element('img');image.src=preview;image.alt='第 '+(i+1)+' 张手动框选局部图';jump.append(image);}else jump.append(element('span','已保存选框'));
        jump.addEventListener('click',()=>{setCapture(false);seekFrame(selection.frame_id);restorePreview(selection);});
        const meta=element('div',null,'llm-evidence-meta');meta.append(element('span',selection.timestamp.toFixed(2)+' s'));
        const remove=element('button','×');remove.type='button';remove.disabled=busy||Boolean(reviewId);remove.setAttribute('aria-label','删除第 '+(i+1)+' 张并重新框选');remove.addEventListener('click',()=>{previews.delete(selection.frame_id);selections.splice(i,1);saveDraft();renderEvidence();dirty=true;});meta.append(remove);slot.append(jump,meta);
      }else{slot.append(element('span',String(i+1),'llm-slot-number'),element('small','等待框选'));}
      slots.push(slot);
    }
    $('llmEvidence').replaceChildren(...slots);controls();
  }
  function applySettings(data){
    settings=data.settings||data;$('llmBaseURL').value=settings.base_url||'';$('llmModel').value=settings.model||'';$('llmAPIKey').value='';
    $('llmAPIKey').placeholder=settings.has_api_key?'已配置，留空保留现有密钥':'未配置密钥';$('llmKeyStorage').value=settings.key_storage==='protected'?'protected':'session';
    $('llmKeyStatus').textContent=(settings.has_api_key?(settings.key_storage==='protected'?'密钥已由 Windows 加密保存在本机。':'密钥仅在本次服务运行期间可用；服务重启后需重新填写。'):'尚未配置 API Key。')+(settings.storage_notice?' '+settings.storage_notice:'');
    $('llmSettingsClear').disabled=!settings.has_api_key||settingsBusy;
    $('llmDestination').replaceChildren(element('small','手动发送目的地'),element('b',settings.endpoint||settings.base_url||'尚未填写 API 地址'),element('span','模型：'+(settings.model||'尚未选择')));
    controls();
  }
  function renderLabels(){
    const active=[...llmLabelRows.values()].filter(row=>row.active!==false);$('llmLabels').replaceChildren();if(!active.length)return;
    $('llmLabels').append(element('small','已确认的物体知识'));
    for(const row of active){const entry=element('div',null,'llm-label-entry'),copy=element('div');copy.append(element('b',row.name),element('small',row.track_id==null?'补充漏检 · 仅视频定位':'关联原记录 #'+row.track_id));
      const revert=element('button','撤销');revert.type='button';revert.disabled=busy;revert.addEventListener('click',async()=>{
        if(busy)return;busy=true;controls();renderLabels();
        try{await request('/api/llm/labels/'+encodeURIComponent(row.id)+'/revert',{});await loadLabels();status('已撤销“'+row.name+'”的辅助标注，原模型数据保留。');}
        catch(error){status(error.message);}finally{busy=false;controls();renderLabels();}
      });entry.append(copy,revert);$('llmLabels').append(entry);
    }
  }
  function applyLabels(rows){
    // Remove only this module's display layer and evidence boxes before replaying
    // the persistent labels. Original geometry and raw model records are untouched.
    llmClassOverrides.clear();
    for(let i=objs.length-1;i>=0;i--)if(objs[i].manual_llm_object)objs.splice(i,1);
    for(const [frame,rowsAtFrame] of observationFrames)observationFrames.set(frame,rowsAtFrame.filter(row=>!row.manual_llm_object));
    llmLabelRows.clear();
    for(const row of rows||[]){
      if(row.active===false||typeof row.name!=='string'||!row.name.trim())continue;llmLabelRows.set(String(row.id),row);
      const existing=row.track_id==null?null:objs.find(o=>String(o.track_id)===String(row.track_id));
      const track=existing?.track_id??(row.object_id||'manual_'+row.id),key=String(track);
      llmClassOverrides.set(key,{track_id:track,class_name:row.name,model_class:existing?.class_name||null,source:'manual_llm_assisted',label_id:row.id});
      if(!existing){
        const evidence=(row.selections||[]).filter(s=>Number.isInteger(s.frame_id)&&Array.isArray(s.xyxy));
        const object={track_id:track,class_name:row.name,manual_llm_object:true,manual_selections:evidence,frame_id:evidence.at(-1)?.frame_id??0,observations:evidence.length,confidence:null,xyz:null,position_status:'unavailable',position_source:'none',position_relation_usable:false,depth_source:'none'};
        objs.push(object);for(const selection of evidence){const box={...object,frame_id:selection.frame_id,xyxy:selection.xyxy};if(!observationFrames.has(selection.frame_id))observationFrames.set(selection.frame_id,[]);observationFrames.get(selection.frame_id).push(box);}
      }
    }
    activeClass='';populateClassFilter();populateTargets();populatePairs();renderObjectList();if(selectedTrackId!=null){
      if(objs.some(o=>o.track_id===selectedTrackId))renderDetails();else{selectedTrackId=null;$('selectedTag').textContent='选择一个物体';$('objectDetails').textContent='点选场景标记或下方清单，查看坐标、类别一致性与定位依据。';}
    }renderLabels();dirty=true;
  }
  async function loadLabels(){const data=await request('/api/llm/labels?run_id='+encodeURIComponent(runId));applyLabels(data.labels||[]);}
  function showResult(review){
    const result=review.result||{};reviewId=String(review.id||reviewId);reviewComplete=true;confirmed=false;
    $('llmResponse').hidden=false;$('llmSuggestedName').textContent=result.name||'模型未给出明确名称';$('llmReason').textContent=textValue(result.reason)||'未提供解释。';
    $('llmUncertainty').textContent='仍需核对：'+(textValue(result.uncertainty)||'名称建议需要结合原画面确认。');
    $('llmCandidates').replaceChildren(...(Array.isArray(result.candidates)?result.candidates:[]).map(candidate=>element('span',typeof candidate==='string'?candidate:candidate.name||candidate.class_name||textValue(candidate))));
    sameObjectWarning=result.same_object===false;$('llmSameObjectWarning').hidden=!sameObjectWarning;$('llmSameObjectAck').checked=false;
    $('llmConfirmedName').value=result.name||'';status(sameObjectWarning?'大模型提示三张图可能不是同一物体，请先逐一核对或重新选图。':'大模型已返回建议。请核对并确认名称后再更新结果。');saveDraft();controls();
  }
  function failReview(review){
    // Only an authoritative terminal error releases the task. A failed status
    // request may still refer to a running provider call and must retain its ID.
    reviewId=null;reviewComplete=false;confirmed=false;sameObjectWarning=false;
    $('llmResponse').hidden=true;saveDraft();
    status((review.error||'大模型请求失败。')+' 三张选图已保留，可调整后手动重发。');
  }
  async function pollReview(id){
    const epoch=++pollEpoch;busy=true;controls();renderEvidence();
    try{
      for(let count=0;count<240;count++){
        const review=await request('/api/llm/reviews/'+encodeURIComponent(id));if(epoch!==pollEpoch)return;
        if(review.status==='complete'){showResult(review);return;}
        if(review.status==='error'){failReview(review);return;}
        status('正在等待视觉大模型辨认… 可以继续查看视频，结果将显示在这里。');await new Promise(resolve=>setTimeout(resolve,1000));
      }
      throw new Error('请求等待时间较长。重新打开此结果会继续读取任务状态，不会重复发送图片。');
    }catch(error){status(error.message+' 选图与任务编号已保留，重新打开此结果可继续查询；不会自动重发。');}
    finally{if(epoch===pollEpoch){busy=false;controls();renderEvidence();renderLabels();}}
  }
  async function connect(){
    if(typeof fetch!=='function')return;
    try{
      service=await request('/api/llm/context?run_id='+encodeURIComponent(runId||''));applySettings(service.settings||await request('/api/llm/settings'));await loadLabels();
      let stored=null;try{stored=JSON.parse(window.localStorage?.getItem(draftKey)||'null');}catch{/* A damaged local draft cannot disable the API settings. */}
      if(stored&&Array.isArray(stored.selections)){
        const seen=new Set();selections=stored.selections.filter(s=>Number.isInteger(s.frame_id)&&s.frame_id>=0&&Array.isArray(s.xyxy)&&s.xyxy.length===4&&s.xyxy.every(Number.isFinite)&&s.xyxy[2]>s.xyxy[0]&&s.xyxy[3]>s.xyxy[1]&&!seen.has(s.frame_id)&&Boolean(seen.add(s.frame_id))).slice(0,3).map(s=>({...s,timestamp:Number.isFinite(s.timestamp)?s.timestamp:timeForFrame(s.frame_id)}));
        populateTargets(stored.target);$('llmQuestion').value=String(stored.question||'').slice(0,600);confirmed=Boolean(stored.confirmed);
        if(selections.length){$('llmPanel').hidden=false;for(const selection of selections)restorePreview(selection);}
        if(stored.review_id&&!confirmed){reviewId=String(stored.review_id);pollReview(reviewId);}
      }
      renderEvidence();if(!reviewId)status(settings.has_api_key?'API 已就绪，选满三个时刻后可以手动发送。':'先填写 API 设置，再选取三个时刻。');
    }catch(error){status(error.message);$('llmDestination').textContent=error.message;}
  }
  function sourcePoint(event){
    const rect=overlay.getBoundingClientRect(),w=overlay.clientWidth,h=overlay.clientHeight;
    const scale=Math.min(w/video.videoWidth,h/video.videoHeight),left=(w-video.videoWidth*scale)/2,top=(h-video.videoHeight*scale)/2;
    return [clamp((event.clientX-rect.left-left)/scale,0,video.videoWidth),clamp((event.clientY-rect.top-top)/scale,0,video.videoHeight)];
  }
  $('llmStart').addEventListener('click',()=>{
    $('llmPanel').hidden=false;setPlaying(false);if(!selections.length&&!reviewId)populateTargets(selectedTrackId??'');
    controls();renderEvidence();$('llmPanel').scrollIntoView?.({behavior:'smooth',block:'nearest'});
    if(!service)connect();
  });
  function toggleCapture(){if(busy)return;setPlaying(false);if(!capture&&selections.some(s=>s.frame_id===sourceFrame())){status('此时刻已经选过，请拖动视频时间轴，换一个时刻。');$('llmExpandedCount').textContent='此时刻已选择，请用视频进度条换一个时刻。';return;}expandVideo(true);setCapture(!capture);controls();}
  $('llmCapture').addEventListener('click',toggleCapture);$('llmExpandedCapture').addEventListener('click',toggleCapture);
  $('llmVideoExpand').addEventListener('click',()=>{$('llmPanel').hidden=false;setPlaying(false);expandVideo(!videoExpanded);controls();});
  $('llmVideoCollapse').addEventListener('click',()=>{expandVideo(false);$('llmPanel').scrollIntoView?.({behavior:'smooth',block:'nearest'});});
  overlay.addEventListener('pointerdown',event=>{if(!capture||busy||event.button!==0||video.seeking||video.readyState<2)return;event.preventDefault();setPlaying(false);drag={start:sourcePoint(event),end:sourcePoint(event),frame:sourceFrame(),time:video.currentTime,id:event.pointerId};overlay.setPointerCapture(event.pointerId);dirty=true;});
  overlay.addEventListener('pointermove',event=>{if(!drag||drag.id!==event.pointerId)return;drag.end=sourcePoint(event);dirty=true;});
  overlay.addEventListener('pointerup',event=>{
    if(!drag||drag.id!==event.pointerId)return;const current=drag;current.end=sourcePoint(event);overlay.releasePointerCapture?.(event.pointerId);drag=null;
    if(current.frame!==sourceFrame()){status('画面时刻发生变化，请暂停后重新框选。');setCapture(false);return;}
    const xyxy=[Math.floor(Math.min(current.start[0],current.end[0])),Math.floor(Math.min(current.start[1],current.end[1])),Math.ceil(Math.max(current.start[0],current.end[0])),Math.ceil(Math.max(current.start[1],current.end[1]))];
    if(xyxy[2]-xyxy[0]<8||xyxy[3]-xyxy[1]<8){status('选框太小，请框住物体的完整边界。');dirty=true;return;}
    if(selections.length>=3||selections.some(s=>s.frame_id===current.frame)){status('请换到一个尚未选择的视频时刻。');setCapture(false);return;}
    const selection={frame_id:current.frame,xyxy,timestamp:current.time},preview=previewFrom(video,selection);selections.push(selection);if(preview)previews.set(selection.frame_id,preview);
    setCapture(false);if(selections.length===3){expandVideo(false);$('llmPanel').scrollIntoView?.({behavior:'smooth',block:'nearest'});}saveDraft();renderEvidence();status(selections.length===3?'三张图已选好，请核对发送地址与模型，再手动发送。':'已选 '+selections.length+'/3 张，请换一个时刻继续框选。');dirty=true;
  });
  overlay.addEventListener('pointercancel',()=>setCapture(false));
  video.addEventListener('play',()=>setCapture(false));for(const event of ['loadedmetadata','loadeddata','seeking','seeked'])video.addEventListener(event,controls);
  $('llmReset').addEventListener('click',()=>{if(busy||reviewId&&!reviewComplete)return;pollEpoch++;selections=[];previews.clear();reviewId=null;reviewComplete=false;confirmed=false;setCapture(false);$('llmResponse').hidden=true;saveDraft();renderEvidence();status('选图已清空，可以重新选择。');});
  $('llmTarget').addEventListener('change',saveDraft);$('llmQuestion').addEventListener('input',saveDraft);$('llmConfirmedName').addEventListener('input',controls);$('llmSameObjectAck').addEventListener('change',controls);
  $('llmSubmit').addEventListener('click',async()=>{
    if(busy||reviewId||selections.length!==3||!settings?.has_api_key)return;setCapture(false);busy=true;controls();renderEvidence();status('正在提交三张局部图…');
    const target=$('llmTarget').value,original=objs.find(o=>String(o.track_id)===target);
    try{
      const payload={run_id:runId,selections:selections.map(({frame_id,xyxy})=>({frame_id,xyxy})),question:$('llmQuestion').value.trim()};
      if(service.library_id)payload.library_id=service.library_id;if(target&&original)payload.track_id=original.track_id;
      const review=await request('/api/llm/reviews',payload);reviewId=String(review.id);saveDraft();
      if(review.status==='complete'||review.status==='error'){
        if(review.status==='complete')showResult(review);else failReview(review);
        busy=false;controls();renderEvidence();
      }else await pollReview(reviewId);
    }catch(error){busy=false;status(error.message);controls();renderEvidence();}
  });
  $('llmConfirm').addEventListener('click',async()=>{
    const name=$('llmConfirmedName').value.trim();if(busy||!reviewComplete||confirmed||!name||(sameObjectWarning&&!$('llmSameObjectAck').checked))return;busy=true;controls();
    try{
      const response=await request('/api/llm/reviews/'+encodeURIComponent(reviewId)+'/confirm',{name});confirmed=true;saveDraft();await loadLabels();
      if(response.label){const object=objs.find(o=>String(o.track_id)===String(response.label.track_id??response.label.object_id));if(object)selectObject(object);}
      status('已保存“'+name+'”，物体清单与视频标注已更新。名称知识已记录；不会修改原始模型或凭空生成三维体积。');
    }catch(error){status(error.message);}finally{busy=false;controls();renderLabels();}
  });
  $('llmSettingsOpen').addEventListener('click',async()=>{openDialog('llmSettingsDialog');$('llmSettingsStatus').textContent='';try{if(!service)await connect();applySettings(await request('/api/llm/settings'));}catch(error){$('llmSettingsStatus').textContent=error.message;}});
  $('llmSettingsClose').addEventListener('click',()=>closeDialog('llmSettingsDialog'));
  $('llmSettingsDialog').addEventListener('close',()=>{$('llmAPIKey').value='';});
  $('llmSettingsForm').addEventListener('submit',async event=>{
    event.preventDefault();if(settingsBusy)return;settingsBusy=true;$('llmSettingsSave').disabled=true;
    const payload={base_url:$('llmBaseURL').value.trim(),model:$('llmModel').value.trim(),key_storage:$('llmKeyStorage').value};if($('llmAPIKey').value.trim())payload.api_key=$('llmAPIKey').value.trim();$('llmAPIKey').value='';
    try{applySettings(await request('/api/llm/settings',payload));$('llmSettingsStatus').textContent='设置已保存，只有手动发送选图才调用。';}
    catch(error){$('llmSettingsStatus').textContent=error.message;}finally{settingsBusy=false;$('llmSettingsSave').disabled=false;$('llmSettingsClear').disabled=!settings?.has_api_key;}
  });
  $('llmSettingsClear').addEventListener('click',async()=>{
    if(settingsBusy)return;settingsBusy=true;$('llmSettingsClear').disabled=true;$('llmAPIKey').value='';
    try{applySettings(await request('/api/llm/settings',{clear_api_key:true}));$('llmSettingsStatus').textContent='API Key 已清除。';}
    catch(error){$('llmSettingsStatus').textContent=error.message;}finally{settingsBusy=false;$('llmSettingsClear').disabled=!settings?.has_api_key;}
  });
  window.addEventListener('pagehide',()=>{pollEpoch++;saveDraft();});
  populateTargets();renderEvidence();if(typeof fetch==='function')connect();
  return {drawSelection(g,{scale,left,top}){
    if(!capture&&!selections.length)return;g.save();
    for(const selection of selections.filter(s=>s.frame_id===sourceFrame())){const [x1,y1,x2,y2]=selection.xyxy;g.strokeStyle='#477b70';g.lineWidth=2;g.setLineDash([5,3]);g.strokeRect(left+x1*scale,top+y1*scale,(x2-x1)*scale,(y2-y1)*scale);}
    if(drag){const [x1,y1]=drag.start,[x2,y2]=drag.end;g.setLineDash([]);g.strokeStyle='#477b70';g.fillStyle='#477b7025';g.lineWidth=2;g.fillRect(left+x1*scale,top+y1*scale,(x2-x1)*scale,(y2-y1)*scale);g.strokeRect(left+x1*scale,top+y1*scale,(x2-x1)*scale,(y2-y1)*scale);}
    g.restore();
  },applyLabels,cancelCapture:()=>expandVideo(false),isExpanded:()=>videoExpanded};
})();
let listFrame=-1;
$('timeScope').checked=currentOnly;
renderTrackingCoverage();populatePairs();renderObjectList();updateViewButtons();requestAnimationFrame(loop);
if(hasGeometry&&!hasVideo)showInformativeFrame();
