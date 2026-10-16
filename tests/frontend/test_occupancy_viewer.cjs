'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),test=require('node:test');
const source=fs.readFileSync(path.join(__dirname,'test_viewer.cjs'),'utf8');
const {harness}=Function('require','__dirname',source.slice(0,source.indexOf("test('front projection"))+'\nreturn {harness};')(require,__dirname);
const plain=x=>JSON.parse(JSON.stringify(x));
function fixture(){
  const volumes=[{min:[-1,-1,5],max:[1,1,7]},{min:[2,-1,5],max:[4,1,7]}].map(v=>({...v,
    frame_id:0,world_frame_id:'a',available:true,candidate_overlap_usable:true,size:[2,2,2],volume_map_units3:8,
    status:'inferred_unverified',collision_usable:false}));
  const objects=volumes.map((v,i)=>({track_id:i+1,class_name:'bottle',frame_id:30,xyz:[100+i,100,100],
    confidence:.9,observations:5,occupancy:{...v,frame_id:30,min:[99,99,99],max:[101,101,101]}}));
  const rows=volumes.map((v,i)=>({track_id:i+1,class_name:'bottle',frame_id:0,xyxy:[10,10,20,20],
    geometry:{frame_id:0,xyz:[i*3,0,6],world_frame_id:'a',pose_valid:true,pose_trusted:true,
      temporal_consistency:'consistent',position_relation_usable:true,occupancy:v}}));
  const app=harness({objects,object_observations:rows,video_filename:'source_video.mp4',trajectory_world_frame_ids:['a','a'],
    summary:{scale_consistent:true,world_frame_id:'a',video:{fps:30},frame_quality:{0:{pose_valid:true,world_frame_id:'a'},30:{pose_valid:true,world_frame_id:'a'}}}});
  app.get('occupancyMargin').value='0';return app;
}
test('current volume uses exact-frame occupancy and cannot inherit final volume',()=>{
  const app=fixture();assert.deepEqual(plain(app.run('occupancyFor(objs[0]).min')),[-1,-1,5]);
  app.get('sourceVideo').currentTime=1;assert.equal(app.run('occupancyFor(objs[0])'),null);
});
test('candidate gaps and margin overlap never claim certified collision clearance',()=>{
  const app=fixture();const separated=app.run('occupancyPairState(...objs)');
  assert(separated.usable);assert(!separated.overlap);assert.equal(separated.gap,1);
  assert.equal(separated.a.collision_usable,false);
  app.get('occupancyMargin').value='.6';const buffered=app.run('occupancyPairState(...objs)');
  assert(buffered.overlap);assert(!buffered.raw_overlap);
  app.run('updateOccupancyReadout()');assert(app.get('occupancyReadout').children[0].textContent.includes('潜在交叠'));
});
test('unknown membership, other worlds and missing pose forbid volume comparison',()=>{
  const app=fixture();app.run('observations[0].geometry.occupancy.candidate_overlap_usable=false');
  assert.equal(app.run('occupancyPairState(...objs).usable'),false);
  app.run("observations[0].geometry.occupancy.world_frame_id='other'");
  assert.equal(app.run('occupancyFor(objs[0])'),null);
  app.run("observations[0].geometry.occupancy.world_frame_id='a';DATA.summary.frame_quality['0'].pose_valid=false");
  assert.equal(app.run('occupancyFor(objs[0])'),null);
});
test('gray volume faces are selectable and volume layer can be disabled',()=>{
  const app=fixture();app.run('target=[1.5,0,6];dist=8;yaw=.62;pitch=.3;draw()');
  const hit=app.run('volumeHitTargets.at(-1)');assert(hit);
  const x=hit.polygon.reduce((s,p)=>s+p.x,0)/hit.polygon.length;
  const y=hit.polygon.reduce((s,p)=>s+p.y,0)/hit.polygon.length;
  const picked=app.run(`pickAt(${x},${y})`);assert.equal(picked.trackId,hit.trackId);
  app.get('layerVolumes').events.change({target:{checked:false}});app.run('draw()');
  assert.equal(app.run('volumeHitTargets.length'),0);
});
test('focusing a volume fits all eight corners and preserves calibrated world geometry',()=>{
  const app=fixture();app.run('selectedTrackId=1;focusSelection()');
  assert.deepEqual(plain(app.run('target')),[0,0,6]);
  assert(app.run('occupancyCorners(objs[0]).every(p=>{const q=project(p,viewerBasis(),600,400);return q&&q.x>0&&q.x<600&&q.y>0&&q.y<400;})'));
  assert.deepEqual(plain(app.run('observations[0].geometry.xyz')),[0,0,6]);
});
test('a visibly overlaid position marker remains selectable over another gray face',()=>{
  const app=fixture();app.run('hitTargets=[{trackId:1,x:200,y:100}];volumeHitTargets=[{trackId:2,polygon:[{x:100,y:50},{x:300,y:50},{x:300,y:150},{x:100,y:150}],depth:2}]');
  assert.equal(app.run('pickAt(200,100).trackId'),1);
  assert.equal(app.run('pickAt(150,100).trackId'),2);
});
test('history preview selects only past geometry and never the final future volume',()=>{
  const app=fixture();app.run(`objs[0].position_snapshots=[observations[0].geometry,
    {...observations[0].geometry,frame_id:30,xyz:[100,100,100],occupancy:objs[0].occupancy}];currentOnly=false`);
  assert.deepEqual(plain(app.run('occupancyFor(objs[0]).min')),[-1,-1,5]);
  assert.deepEqual(plain(app.run('positionSnapshot(objs[0]).xyz')),[0,0,6]);
  app.run('objs[0].position_snapshots.shift()');assert.equal(app.run('occupancyFor(objs[0])'),null);
});
