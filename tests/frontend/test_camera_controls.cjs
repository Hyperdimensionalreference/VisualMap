'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const test=require('node:test');
const existing=fs.readFileSync(path.join(__dirname,'test_viewer.cjs'),'utf8');
const prefix=existing.slice(0,existing.indexOf("test('front projection"));
const {harness}=Function('require','__dirname',prefix+'\nreturn {harness};')(require,__dirname);
const plain=value=>JSON.parse(JSON.stringify(value));
const near=(actual,expected,tolerance=1e-6)=>assert.ok(Math.abs(actual-expected)<=tolerance,`${actual} != ${expected}`);
function pointer(button,x,y,id=7){return {button,buttons:button===0?1:button===1?4:2,
  clientX:x,clientY:y,pointerId:id,shiftKey:false,preventDefault(){}};}
const object={track_id:1,class_name:'bottle',confidence:.9,observations:4,frame_id:30,
  xyz:[2,0,4],depth_source:'depth_median',position_source:'depth_median',position_status:'estimated'};

for(const mode of ['perspective','top'])test(`${mode} wheel changes only display size, keeping eye and focus fixed`,()=>{
  const app=harness();app.run(`setView('${mode}');target=[0,0,0];dist=10`);
  const before=plain(app.run('({eye:viewerBasis().eye,target,dist})'));
  const result=app.run(`const anchor=worldAtScreen(445,130);zoomAt(445,130,-160);
    ({point:project(anchor,viewerBasis(),600,400),factor:zoomFactor()})`);
  near(result.point.x,300+145*result.factor,.003);near(result.point.y,200-70*result.factor,.003);
  assert.deepEqual(plain(app.run('({eye:viewerBasis().eye,target,dist})')),before);
});

test('leaving a calibrated rolled camera preserves both the image and camera position',()=>{
  const app=harness({trajectory:[[2,3,4]],trajectory_timestamps:[0],trajectory_frame_ids:[0],
    trajectory_rotations:[[[0,-1,0],[0,0,1],[-1,0,0]]],
    summary:{intrinsics:{fx:100,fy:80,cx:30,cy:20,width:100,height:80,dist:[.1,.01,.02,-.01,.005]}}});
  const result=app.run(`mode='camera';const p=[1,8,3];
    const before=project(p,viewerBasis(),600,400);leaveCameraMode();
    ({before,after:project(p,viewerBasis(),600,400),eye:viewerBasis().eye,mode})`);
  near(result.after.x,result.before.x);near(result.after.y,result.before.y);
  result.eye.forEach((v,i)=>near(v,[2,3,4][i]));assert.equal(result.mode,'perspective');
});

for(const button of [1,2])test(`${button===1?'middle':'right'} button pans without rotating`,()=>{
  const app=harness();const before=plain(app.run('({yaw,pitch,target})'));
  app.get('view').events.pointerdown(pointer(button,100,100));
  app.get('view').events.pointermove(pointer(button,130,115));
  const after=plain(app.run('({yaw,pitch,target})'));
  near(after.yaw,before.yaw);near(after.pitch,before.pitch);assert.notDeepEqual(after.target,before.target);
});

test('right click never selects a target and a tiny left movement does not exit camera mode',()=>{
  const app=harness({objects:[object]});app.run("mode='camera';hitTargets=[{trackId:1,x:100,y:100}]");
  app.get('view').events.pointerdown(pointer(2,100,100));
  app.get('view').events.pointerup(pointer(2,100,100));
  assert.equal(app.run('selectedTrackId'),null);
  app.get('view').events.pointerdown(pointer(0,100,100));
  app.get('view').events.pointermove(pointer(0,101,101));
  assert.equal(app.run('mode'),'camera');
});

test('selection preserves video time, zoom, orbit target and following',()=>{
  const app=harness({objects:[object]});app.run('follow=true;tNorm=.3;dist=8;target=[1,2,3]');
  const before=plain(app.run('({tNorm,dist,target,yaw,pitch,follow})'));
  app.run('selectObject(objs[0])');
  assert.deepEqual(plain(app.run('({tNorm,dist,target,yaw,pitch,follow})')),before);
  assert.equal(app.run('selectedTrackId'),1);
});

test('lost pointer capture clears the drag, so later movement cannot move the camera',()=>{
  const app=harness();app.get('view').events.pointerdown(pointer(0,100,100));
  app.get('view').events.lostpointercapture(pointer(0,100,100));
  const before=plain(app.run('cameraValues()'));
  app.get('view').events.pointermove(pointer(0,160,100));
  assert.deepEqual(plain(app.run('cameraValues()')),before);
});

test('fitting rejects a remote cloud outlier without deleting source map points',()=>{
  const points=Array.from({length:100},(_,i)=>[Math.sin(i),Math.cos(i),i/100]);points.push([1e6,1e6,1e6]);
  const app=harness({map_points:points});app.get('fitBtn').events.click();
  assert.ok(app.run('dist')<20);assert.equal(app.run('pts.length'),101);
  assert.ok(app.run('scene.radius')<3);
});

test('automatic camera motion is a closed time function and reduced motion lands immediately',()=>{
  const a=harness(),b=harness();
  for(const app of [a,b]){
    app.context.window.matchMedia=()=>({matches:false});
    app.run('moveCamera({target:[4,3,2],dist:8,yaw:1,pitch:.2})');
  }
  a.run('sampleCameraMotion(.2)');b.run('sampleCameraMotion(.1);sampleCameraMotion(.2)');
  const av=plain(a.run('cameraValues()')),bv=plain(b.run('cameraValues()'));
  av.forEach((v,i)=>near(v,bv[i]));
  a.context.window.matchMedia=()=>({matches:true});
  a.run('moveCamera({target:[9,8,7],dist:3})');
  assert.deepEqual(plain(a.run('target')),[9,8,7]);assert.equal(a.run('cameraMotion'),null);
});

test('repeat zoom-in / zoom-out at the same cursor remains reversible for 500 cycles',()=>{
  const app=harness();app.run('target=[0,0,.5];dist=4;yaw=.3;pitch=.2');
  const before=plain(app.run('cameraValues()'));
  app.run('for(let i=0;i<500;i++){zoomAt(510,75,-230);zoomAt(510,75,230);}');
  plain(app.run('cameraValues()')).forEach((value,i)=>near(value,before[i],1e-8));
});

test('free navigation sheds a distorted calibrated lens before repeated zoom and pan',()=>{
  const app=harness({objects:[object],trajectory_rotations:[[[1,0,0],[0,1,0],[0,0,1]]],
    summary:{intrinsics:{fx:100,fy:80,cx:30,cy:20,width:100,height:80,dist:[.4,.15,.02,-.01,.1]}}});
  app.run("mode='camera';zoomAt(510,60,-150)");
  assert.equal(app.run('freeLens'),null);assert.equal(app.run('lensMix'),0);
  for(let i=0;i<120;i++){
    app.run(`zoomAt(${i%2?510:30},${i%3?30:350},${i%2?400:-400})`);
    const button=i%2?1:2;
    app.get('view').events.pointerdown(pointer(button,60,80,i+1));
    app.get('view').events.pointermove(pointer(button,500,330,i+1));
    app.get('view').events.pointerup(pointer(button,500,330,i+1));
    assert(app.run('cameraValues().every(Number.isFinite)'));
    assert(app.run('dist>=cameraLimits().min&&dist<=cameraLimits().max'));
    assert(app.run('Math.hypot(...sub(target,scene.center))<=scene.radius*120+1e-8'));
  }
});

test('full panorama includes sparse valid poses and objects rather than filtering them by cloud density',()=>{
  const points=Array.from({length:120},(_,i)=>[Math.sin(i),Math.cos(i),Math.sin(i*.3)]);
  const app=harness({trajectory:[[0,0,0],[80,5,12]],map_points:points,objects:[{...object,xyz:[-40,-4,18]}]});
  app.run('fit(fittedScenePoints(),true)');
  assert.deepEqual(plain(app.run('target')),plain(app.run('bounds(fittedScenePoints()).center')));
  assert(app.run('fittedScenePoints().every(p=>{const q=project(p,viewerBasis(),600,400);return q&&q.x>0&&q.x<600&&q.y>0&&q.y<400;})'));
});

test('reset cancels a drag, lens and camera transition without resetting selection or video time',()=>{
  const app=harness({objects:[object]});app.context.window.matchMedia=()=>({matches:false});
  app.run("selectedTrackId=1;tNorm=.37;setView('camera');sampleCameraMotion(.15)");
  app.get('view').events.pointerdown(pointer(1,100,100));
  app.get('resetBtn').events.click();app.run('sampleCameraMotion(1)');
  assert.equal(app.run('selectedTrackId'),1);near(app.run('tNorm'),.37);
  assert.equal(app.run('drag'),null);assert.equal(app.run('freeLens'),null);
  assert.equal(app.run('pendingMode'),null);assert.equal(app.run('cameraMotion'),null);
  assert.equal(app.run('mode'),'perspective');assert.equal(app.run('lensMix'),0);
  assert.deepEqual(plain(app.run('cameraUp')),[0,-1,0]);
  const before=plain(app.run('cameraValues()'));
  app.get('view').events.pointermove(pointer(1,300,300));assert.deepEqual(plain(app.run('cameraValues()')),before);
});

test('recovery clears previous velocity and remains stable after invalid camera inputs',()=>{
  const app=harness();app.context.window.matchMedia=()=>({matches:false});
  app.run('moveCamera({target:[20,20,20],dist:12});sampleCameraMotion(.2);fit(fittedScenePoints(),true)');
  assert(app.run('cameraMotion.velocity.every(v=>v===0)'));
  app.run('target=[NaN,Infinity,-Infinity];dist=NaN;yaw=NaN;pitch=NaN;cameraUp=[0,0,0];fit(fittedScenePoints(),true);sampleCameraMotion(1)');
  assert(app.run('cameraValues().every(Number.isFinite)'));
});

test('undo restores camera only, and extreme repeated rolled orbit has an orthonormal basis',()=>{
  const app=harness({objects:[object]});app.run('tNorm=.61;selectedTrackId=1');
  const before=plain(app.run('cameraValues()'));app.run("setView('side');undoCamera()");
  assert.deepEqual(plain(app.run('cameraValues()')),before);near(app.run('tNorm'),.61);assert.equal(app.run('selectedTrackId'),1);
  app.run('cameraUp=[0,0,-1]');
  for(let i=0;i<80;i++){
    app.get('view').events.pointerdown(pointer(0,200,150,i+1));
    app.get('view').events.pointermove(pointer(0,570,350,i+1));
    app.get('view').events.pointerup(pointer(0,570,350,i+1));
    const result=app.run('(()=>{const basis=viewerBasis();return [Math.hypot(...basis.x),Math.hypot(...basis.y),Math.hypot(...basis.z),dot(basis.x,basis.y),dot(basis.y,basis.z),dot(basis.x,basis.z)];})()');
    result.forEach((value,j)=>near(value,j<3?1:0,1e-8));
  }
});

test('one large wheel event and several reverse events return identical projected size',()=>{
  const app=harness();app.run('target=[0,0,4];dist=10');
  const before=plain(app.run('project([1,1,4],viewerBasis(),600,400)'));
  app.run('zoomAt(550,40,800);for(let i=0;i<4;i++)zoomAt(30,350,-200)');
  near(app.run('zoomFactor()'),1,1e-12);
  const after=plain(app.run('project([1,1,4],viewerBasis(),600,400)'));
  for(const key of ['x','y','z','scale'])near(after[key],before[key],1e-9);
});

test('changing the cursor on the reverse zoom cannot pan the scene out of view',()=>{
  const app=harness();const before=plain(app.run('cameraValues()'));
  app.run('for(let i=0;i<5;i++)zoomAt(560,30,240);for(let i=0;i<5;i++)zoomAt(30,360,-240)');
  plain(app.run('cameraValues()')).forEach((v,i)=>near(v,before[i],1e-10));
});

test('large outward and inward wheel gestures remain reversible without crossing landmarks',()=>{
  const app=harness();const before=plain(app.run('viewerBasis().eye'));
  app.run('for(let i=0;i<35;i++)zoomAt(580,10,240)');
  assert(app.run('zoomFactor()>0&&zoomFactor()<1'));
  assert.deepEqual(plain(app.run('viewerBasis().eye')),before);
  app.run('for(let i=0;i<35;i++)zoomAt(20,390,-240)');
  near(app.run('zoomFactor()'),1,1e-11);
  assert.deepEqual(plain(app.run('viewerBasis().eye')),before);
});

test('closed rectangular orbit paths restore screen points without accumulating roll',()=>{
  const app=harness();app.run('target=[0,0,4];dist=12;yaw=.62;pitch=.25;cameraUp=[0,-1,0]');
  const before=plain(app.run('[[1,1,4],[-1,0,3],[0,-2,5]].map(p=>project(p,viewerBasis(),600,400))'));
  for(let cycle=0;cycle<10;cycle++){
    app.get('view').events.pointerdown(pointer(0,100,100));
    for(const [x,y] of [[240,100],[240,160],[100,160],[100,100]])
      app.get('view').events.pointermove(pointer(0,x,y));
    app.get('view').events.pointerup(pointer(0,100,100));
  }
  assert.deepEqual(plain(app.run('cameraUp')),[0,-1,0]);
  const after=plain(app.run('[[1,1,4],[-1,0,3],[0,-2,5]].map(p=>project(p,viewerBasis(),600,400))'));
  after.forEach((p,i)=>{near(p.x,before[i].x,1e-8);near(p.y,before[i].y,1e-8);});
});

for(const mode of ['top','front','side'])test(`${mode} rotation has no release-time projection or size change`,()=>{
  const app=harness();app.run(`setView('${mode}')`);
  app.get('view').events.pointerdown(pointer(0,100,100));
  app.get('view').events.pointermove(pointer(0,140,80));
  const before=plain(app.run('({values:cameraValues(),p:project([.2,.1,.4],viewerBasis(),600,400)})'));
  app.get('view').events.pointerup(pointer(0,140,80));app.run('sampleCameraMotion(99)');
  assert.deepEqual(plain(app.run('({values:cameraValues(),p:project([.2,.1,.4],viewerBasis(),600,400)})')),before);
  assert.equal(app.run('mode'),'orthographic');assert.equal(app.run('cameraMotion'),null);
});

test('interrupting a preset with a wheel leaves a truthful single projection',()=>{
  const app=harness();app.context.performance={now:()=>120};
  app.context.window.matchMedia=()=>({matches:false});
  app.run("setView('front');sampleCameraMotion(.24);zoomAt(500,20,120)");
  assert.equal(app.run('projectionMix'),0);assert.equal(app.run('mode'),'orthographic');
  assert.equal(app.run('cameraMotion'),null);
  const before=plain(app.run('cameraValues()'));app.run('sampleCameraMotion(10)');
  assert.deepEqual(plain(app.run('cameraValues()')),before);
});

test('direction buttons return the same framing after zoom, pan and orbit',()=>{
  const app=harness({map_points:[[0,0,0],[2,1,4],[-1,-1,2]],objects:[object]});
  for(const mode of ['perspective','top','front','side']){
    app.run(`setView('${mode}')`);
    const before=plain(app.run('fittedScenePoints().map(p=>project(p,viewerBasis(),600,400))'));
    app.run('target=[100,50,-100];zoomAt(30,30,-800);yaw+=.7;pitch=.3');
    app.run(`setView('${mode}')`);
    const after=plain(app.run('fittedScenePoints().map(p=>project(p,viewerBasis(),600,400))'));
    after.forEach((p,i)=>{assert(p&&p.x>0&&p.x<600&&p.y>0&&p.y<400);near(p.x,before[i].x);near(p.y,before[i].y);});
    near(app.run('zoomFactor()'),1);
  }
});

test('size reset and slider ends keep angle, eye and selection and allow immediate wheel changes',()=>{
  const app=harness({objects:[object]});app.run('selectedTrackId=1;tNorm=.41');
  const geometry=plain(app.run('({eye:viewerBasis().eye,target,yaw,pitch})'));
  app.get('zoomRange').events.input({target:{value:'1000'}});
  const large=app.run('zoomFactor()');app.run('zoomAt(100,100,240)');
  assert(app.run('zoomFactor()')<large);
  app.get('zoomReset').events.click();near(app.run('zoomFactor()'),1);
  assert.deepEqual(plain(app.run('({eye:viewerBasis().eye,target,yaw,pitch})')),geometry);
  assert.equal(app.run('selectedTrackId'),1);near(app.run('tNorm'),.41);
});

test('calibrated source-camera zoom returns exact framing without changing its pose or projection',()=>{
  const app=harness();app.run("setView('camera')");
  const before=plain(app.run('({eye:viewerBasis().eye,p:project([1,1,5],viewerBasis(),600,400)})'));
  app.run('zoomAt(580,20,800);zoomAt(20,390,-400);zoomAt(200,300,-400)');
  assert.deepEqual(plain(app.run('({eye:viewerBasis().eye,p:project([1,1,5],viewerBasis(),600,400)})')),before);
  assert.equal(app.run('mode'),'camera');
});

test('rapid opposite fine-angle clicks restore the requested angle instead of the animation midpoint',()=>{
  const app=harness();app.context.window.matchMedia=()=>({matches:false});
  let milliseconds=0;app.context.performance={now:()=>milliseconds};
  const before=plain(app.run('({yaw,pitch,target,dist})'));
  app.run('nudgeView(1,0);sampleCameraMotion(.1)');milliseconds=100;
  app.run('nudgeView(-1,0);sampleCameraMotion(1.1)');
  const after=plain(app.run('({yaw,pitch,target,dist})'));
  near(after.yaw,before.yaw);near(after.pitch,before.pitch);
  assert.deepEqual(after.target,before.target);near(after.dist,before.dist);
});

test('interrupting panorama recovery from an orthographic view cannot leave mixed projection',()=>{
  const app=harness();app.run("setView('front')");
  app.context.window.matchMedia=()=>({matches:false});
  app.context.performance={now:()=>120};
  app.run('fit(fittedScenePoints(),true);sampleCameraMotion(.24);zoomAt(40,300,100)');
  assert.equal(app.run('projectionMix'),1);assert.equal(app.run('mode'),'perspective');
  assert.equal(app.run('cameraMotion'),null);
});
