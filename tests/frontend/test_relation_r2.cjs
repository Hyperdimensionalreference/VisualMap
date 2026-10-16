'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname,
  '../../src/visualmap/visualization/viewer.js'), 'utf8');

function element(tagName = 'div') {
  const el = {tagName, textContent: '', children: [], style: {}, attributes: {},
    events: {}, options: [], value: '', hidden: false, checked: true, disabled: false,
    width: 600, height: 400, clientWidth: 600, clientHeight: 400,
    currentTime: 0, duration: 2, paused: true, ended: false,
    videoWidth: 100, videoHeight: 80, classList: {toggle() {}},
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; this.textContent = ''; },
    add(option) { this.options.push(option); if (this.options.length === 1) this.value = option.value; },
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, callback) { this.events[name] = callback; },
    setPointerCapture() {}, releasePointerCapture() {},
    getBoundingClientRect() { return {left: 0, top: 0}; },
    play() { this.paused = false; return Promise.resolve(); },
    pause() { this.paused = true; },
  };
  let selectedIndex = 0;
  Object.defineProperty(el, 'selectedIndex', {
    get: () => selectedIndex,
    set(index) { selectedIndex = index; el.value = el.options[index]?.value ?? ''; },
  });
  const context = {calls: [], measureText: text => ({width: String(text).length * 7}),
    createLinearGradient: () => ({addColorStop() {}})};
  for (const method of ['setTransform', 'fillRect', 'clearRect', 'strokeRect', 'beginPath',
    'arc', 'stroke', 'fill', 'moveTo', 'lineTo', 'setLineDash', 'fillText']) {
    context[method] = (...args) => context.calls.push({method, args});
  }
  el.getContext = () => context;
  return el;
}

function fixture() {
  const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const yaw90 = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]];
  const geometry = (xyz, camera_xyz) => ({frame_id: 30, timestamp: 1,
    xyz, camera_xyz, position_frame: 30, position_source: 'object_center_multiview',
    status: 'estimated', uncertainty: 0.05, temporal_consistency: 'consistent',
    position_relation_usable: true, pose_valid: true, pose_trusted: true,
    world_frame_id: 'room-a', reprojection: {frame_id: 30, inside_box: true}});
  const makeObject = (track_id, class_name, xyz) => ({track_id, class_name,
    confidence: 0.9, observations: 6, frame_id: 30, xyz,
    position_frame: 30, position_status: 'estimated', depth_source: 'object_center_multiview',
    position_source: 'object_center_multiview', position_uncertainty: 0.05,
    position_temporal_consistency: 'consistent', position_relation_usable: true,
    pose_valid: true, pose_trusted: true, world_frame_id: 'room-a',
    position_history: [{frame_id: 30, xyz, status: 'estimated', uncertainty: 0.05}],
  });
  return {trajectory: [[0, 0, 0], [0, 0, 0]],
    trajectory_timestamps: [0, 1], trajectory_frame_ids: [0, 30],
    trajectory_rotations: [identity, yaw90], trajectory_world_frame_ids: ['room-a', 'room-a'],
    map_points: [], objects: [makeObject(1, 'bottle', [5, 0, 0]),
      makeObject(2, 'chair', [5, 0, -2])],
    object_observations: [
      {frame_id: 30, timestamp: 1, track_id: 1, class_name: 'bottle', confidence: 0.9,
        xyxy: [40, 30, 60, 50], hits: 6, geometry: geometry([5, 0, 0], [0, 0, 5])},
      {frame_id: 30, timestamp: 1, track_id: 2, class_name: 'chair', confidence: 0.9,
        xyxy: [80, 30, 99, 50], hits: 6, geometry: geometry([5, 0, -2], [2, 0, 5])},
    ], summary: {detector: 'yolox_onnx', video: {fps: 30}, scale_consistent: true,
      world_frame_id: 'room-a', frame_quality: {'0': {pose_valid: true, world_frame_id: 'room-a'},
        '30': {pose_valid: true, world_frame_id: 'room-a'}},
      intrinsics: {fx: 100, fy: 100, cx: 50, cy: 40, width: 100, height: 80}},
  };
}

function harness(data = fixture()) {
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) {
      const el = element(id);
      if (id === 'relationBasis') el.value = 'current';
      elements.set(id, el);
    }
    return elements.get(id);
  };
  const context = vm.createContext({DATA: data, document: {getElementById: get,
    createElement: element, createTextNode: text => ({textContent: String(text), children: []})},
    window: {devicePixelRatio: 1, addEventListener() {}},
    Option: function Option(text, value) { this.text = text; this.value = value; },
    requestAnimationFrame() {}, console});
  vm.runInContext(source, context, {filename: 'viewer.js'});
  const run = js => vm.runInContext(js, context);
  run('currentOnly=true; tNorm=1; updateRelation()');
  return {get, run, context};
}

const textOf = el => [el.textContent, ...(el.children || []).map(textOf)].filter(Boolean).join(' ');
const plain = value => JSON.parse(JSON.stringify(value));
const metricsOf = el => (el.children || []).filter(node => node.className === 'metrics');

test('microsecond-rounded video seek still identifies the requested camera frame', () => {
  const data=fixture();
  data.video_filename='source_video.mp4';
  data.trajectory_frame_ids=[0,68];
  data.summary.video.fps=30.00545553737043;
  data.trajectory_timestamps=[0,68/data.summary.video.fps];
  const app=harness(data);
  app.get('sourceVideo').currentTime=2.266254;
  app.run('tNorm=timeToNorm(video.currentTime)');
  assert.equal(app.run('sourceFrame()'),68);
  assert.equal(app.run('currentFrame()'),68);
});

test('source camera view projects landmarks onto the matching video pixel after yaw', () => {
  const app = harness();
  const projected = app.run(`
    mode='camera';
    const cameraPose=poseAt();
    const videoBasis={eye:cameraPose.p,x:cameraPose.R.map(r=>r[0]),
      y:cameraPose.R.map(r=>-r[1]),z:cameraPose.R.map(r=>r[2])};
    [project([5,0,0],videoBasis,600,400),project([5,0,-2],videoBasis,600,400)]
  `);
  assert.equal(projected[0].x, 300);
  assert.equal(projected[0].y, 200);
  assert.equal(projected[1].x, 500);
  assert.equal(projected[1].y, 200);
});

test('initial evidence navigation uses a verified co-observed frame and its selected pair', () => {
  const app = harness();
  app.run('tNorm=0;showInformativeFrame()');
  assert.equal(app.run('currentFrame()'), 30);
  assert.equal(app.get('timeScope').checked, true);
  assert.deepEqual(plain(app.run('pairObjects().map(o=>o.track_id)')), [1, 2]);
  assert.equal(app.run('relationState(...pairObjects()).usable'), true);
});

test('a quarter-turn camera reports right in the current view and behind on the world axes', () => {
  const app = harness();
  const current = app.run("relationState(objs[0],objs[1],'current')");
  assert.equal(current.usable, true);
  assert.deepEqual(plain(current.delta), [2, 0, 0]);
  assert.equal(current.distance, 2);
  app.get('relationBasis').value = 'current';
  app.run('updateRelation()');
  assert.match(textOf(app.get('relationReadout')), /右侧/,
    'after yaw90, world -Z is the current camera right direction');
  app.get('relationBasis').value = 'world';
  assert.deepEqual(plain(app.run("relationState(objs[0],objs[1],'world').delta")), [0, 0, -2]);
  app.run('updateRelation()');
  assert.match(textOf(app.get('relationReadout')), /后方/,
    'the unchanged world displacement is negative Z');
});

test('current geometry comes from its observation even when future history disagrees', () => {
  const data = fixture();
  data.objects[0].xyz = [50, 20, 30];
  data.objects[0].frame_id = 60;
  data.objects[0].position_frame = 60;
  data.objects[0].position_history = [{frame_id: 60, xyz: [50, 20, 30],
    status: 'estimated', uncertainty: 100}];
  const app = harness(data);
  const current = app.run('positionSnapshot(objs[0])');
  assert.deepEqual(plain(current.xyz), [5, 0, 0]);
  assert.equal(current.position_frame, 30);
  assert.equal(current.position_uncertainty, 0.05);
});

test('a geometry snapshot from another frame cannot create a distance line', () => {
  const data = fixture();
  data.object_observations[1].geometry.frame_id = 0;
  data.object_observations[1].geometry.position_frame = 0;
  const app = harness(data);
  assert.equal(app.run('relationState(objs[0],objs[1]).usable'), false);
  assert.equal(metricsOf(app.get('relationReadout')).length, 0);
  const lines = app.run(`
    const testedLines=[];
    line=(a,b)=>testedLines.push([a,b]);
    drawPair(lookAtBasis([0,0,-10],[0,0,0]),600,400);
    testedLines
  `);
  assert.equal(lines.length, 0);
});

test('objects from different reconstructed worlds cannot receive a numeric spatial relation', () => {
  const data = fixture();
  data.object_observations[1].geometry.world_frame_id = 'room-b';
  const app = harness(data);
  assert.equal(app.run('relationState(objs[0],objs[1]).usable'), false);
  assert.equal(metricsOf(app.get('relationReadout')).length, 0);
  assert.match(textOf(app.get('relationReadout')), /坐标|世界|不同/);
});

test('a camera rotation from another reconstructed world cannot explain the current direction', () => {
  const data = fixture();
  for (const row of data.object_observations) row.geometry.world_frame_id = 'room-b';
  const app = harness(data);
  assert.equal(app.run("relationState(objs[0],objs[1],'current').usable"), false,
    'pair coordinates and the camera rotation must belong to the same world');
});

test('an invalid current camera pose blocks spatial distance even with inherited coordinates', () => {
  const data = fixture();
  data.object_observations[1].geometry.pose_valid = false;
  const app = harness(data);
  assert.equal(app.run('relationState(objs[0],objs[1]).usable'), false);
  assert.equal(metricsOf(app.get('relationReadout')).length, 0);
});

test('lost tracking in source-frame quality blocks otherwise accepted object coordinates', () => {
  const data = fixture();
  data.summary.frame_quality['30'].pose_valid = false;
  const app = harness(data);
  assert.equal(app.run('relationState(objs[0],objs[1]).usable'), false);
  assert.equal(metricsOf(app.get('relationReadout')).length, 0);
});

test('unresolved monocular scale blocks distances even when both points look consistent', () => {
  const data = fixture();
  data.summary.scale_consistent = false;
  const app = harness(data);
  assert.equal(app.run('relationState(objs[0],objs[1]).usable'), false);
  assert.equal(metricsOf(app.get('relationReadout')).length, 0);
});

test('an unsampled source frame cannot borrow a neighbouring camera rotation', () => {
  const data = fixture();
  data.video_filename = 'source_video.mp4';
  for (const row of data.object_observations) {
    row.frame_id = 15;
    row.geometry.frame_id = 15;
    row.geometry.position_frame = 15;
  }
  data.summary.frame_quality['15'] = {pose_valid: true, world_frame_id: 'room-a'};
  const app = harness(data);
  app.get('sourceVideo').currentTime = 0.5;
  app.run('tNorm=0.5;updateRelation()');
  assert.equal(app.run('sourceFrame()'), 15);
  assert.equal(app.run('relationState(objs[0],objs[1]).usable'), false);
  assert.equal(metricsOf(app.get('relationReadout')).length, 0);
});

test('screen left/right follows box centres while unverified depth stays undecided', () => {
  const data = fixture();
  for (const row of data.object_observations) row.geometry = {frame_id: 30,
    xyz: null, camera_xyz: null, position_relation_usable: false,
    pose_valid: false, pose_trusted: false, world_frame_id: null};
  const app = harness(data);
  assert.equal(app.run('relationState(objs[0],objs[1]).usable'), false);
  const relation = textOf(app.get('relationReadout'));
  assert.match(relation, /画面|屏幕/);
  assert.match(relation, /右侧/);
  assert.equal(metricsOf(app.get('relationReadout')).length, 0,
    '2D boxes establish image ordering, not a 3D distance');
});
