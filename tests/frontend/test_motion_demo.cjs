'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const motion = require('../../src/visualmap/visualization/viewer_motion.js');
const source = fs.readFileSync(require.resolve('../../scripts/motion_demo.js'), 'utf8');

function fixture(metadata = {}) {
  const elements = new Map();
  const frames = new Map();let frameId = 0;
  const element = () => ({style: {}, dataset: {}, value: '', textContent: '', hidden: false,
    children: [], events: {}, attributes: {}, innerHTML: '', src: '',
    append(node) {this.children.push(node);},
    setAttribute(key, value) {this.attributes[key] = value;},
    addEventListener(key, value) {this.events[key] = value;},
    querySelector() {return element();},
    querySelectorAll() {return [element(), element(), element()];},
    pause() {}, play() {return Promise.resolve();},
  });
  const document = {activeElement: null,
    createElement: element,
    getElementById(id) {if (!elements.has(id)) elements.set(id, element()); return elements.get(id);},
    querySelectorAll() {return Array.from({length: 28}, element);},
    addEventListener() {},
  };
  const window = {VisualMapMotion: motion, MOTION_METADATA: metadata,
    innerWidth: 1440, innerHeight: 1440,
    matchMedia() {return {matches: false, addEventListener() {}};}, addEventListener() {}};
  vm.runInNewContext(source, {window, document, performance: {now: () => 0},
    requestAnimationFrame: callback => {frames.set(++frameId,callback);return frameId;},
    cancelAnimationFrame: id => frames.delete(id), console});
  return {demo: window.motionDemo, elements,flushFrame(now){
    const pending=[...frames.values()];frames.clear();for(const callback of pending)callback(now);
  }};
}

test('seeking during a released drag cancels its old callback without changing the new state',()=>{
  const f=fixture();f.demo.seek(6.22);
  const track={dataset:{drag:'volume'},getBoundingClientRect:()=>({left:0,width:440}),
    setPointerCapture(){},releasePointerCapture(){}};
  const content=f.elements.get('demoContent');
  content.events.pointerdown({target:{closest:()=>track},button:0,pointerId:1,
    clientX:510,preventDefault(){}});
  content.events.pointerup({pointerId:1});
  f.demo.seek(8.72);assert.doesNotThrow(()=>f.flushFrame(120));
  assert.equal(f.demo.currentTime,8.719999999999999);
});

test('demo has 28 beats, 14 seconds, and the requested sequence', () => {
  const {demo} = fixture();
  assert.equal(demo.duration, 14);
  assert.equal(demo.beat, 0.5);
  assert.equal(demo.definitions.length, 28);
  assert.equal(demo.definitions[0][0], 'selection');
  assert.equal(demo.definitions[3][0], 'processing');
  assert.equal(demo.definitions[5][0], 'complete');
  assert.equal(demo.definitions[6][0], 'island');
  assert.equal(demo.definitions[7][0], 'player');
  assert.equal(demo.definitions[12][0], 'volume');
  assert.equal(demo.definitions[15][0], 'toggle');
  assert.equal(demo.definitions[16][0], 'tabs');
  assert.equal(demo.definitions[19][0], 'chart');
  assert.equal(demo.definitions[22][0], 'command');
  assert.equal(demo.definitions[25][0], 'toast');
  assert.equal(demo.definitions[27][0], 'selection');
});

test('arbitrary seek order does not alter any choreography value', () => {
  const {demo} = fixture();
  const times = [0, 1.5, 4.2, 5.1, 6.8, 8.8, 10.7, 11.7, 13.8];
  const models = times.map(t => JSON.stringify(demo.seek(t)));
  times.slice().reverse().forEach(t => demo.seek(t));
  times.forEach((t, i) => assert.equal(JSON.stringify(demo.seek(t)), models[i]));
});

test('loop endpoints are exactly identical including shape, content, and cursor', () => {
  const {demo, elements} = fixture();
  const first = JSON.stringify(demo.seek(0));
  const styles = JSON.stringify([...elements].map(([key, el]) => [key, el.style, el.value, el.textContent]));
  demo.seek(9.8);
  assert.equal(JSON.stringify(demo.seek(14)), first);
  assert.equal(JSON.stringify([...elements].map(([key, el]) => [key, el.style, el.value, el.textContent])), styles);
  assert.equal(JSON.stringify(demo.pureModel(28)), first);
});

test('cursor loop has near-zero speed on both sides of the boundary', () => {
  const {demo} = fixture();
  const first = demo.pureModel(0).cursor, end = demo.pureModel(14 - 1e-5).cursor;
  const next = demo.pureModel(1e-5).cursor;
  assert.ok(Math.hypot(first.x - end.x, first.y - end.y) < 1e-6);
  assert.ok(Math.hypot(first.x - next.x, first.y - next.y) < 1e-6);
});

test('the real approach to the seam closes surface, camera, and hidden controls with zero slope', () => {
  const {demo} = fixture();
  const first = demo.pureModel(0), end = demo.pureModel(14 - 1e-5);
  for (const key of ['width', 'height', 'radius', 'accent', 'contentOpacity', 'blur']) {
    assert.ok(Math.abs(first.shape[key] - end.shape[key]) < 1e-6, key);
  }
  for (const key of ['zoom', 'volume', 'progress', 'chart']) {
    assert.ok(Math.abs(first[key] - end[key]) < 1e-6, key);
  }
  assert.equal(first.shape.label, end.shape.label);
  assert.equal(first.shape.contentState, end.shape.contentState);
  const dt = 1e-4;
  const before = demo.pureModel(14 - dt), after = demo.pureModel(dt);
  assert.ok(Math.abs((end.shape.width - before.shape.width) / dt) < 0.001);
  assert.ok(Math.abs((after.shape.width - first.shape.width) / dt) < 0.001);
});

test('scripted scrubbing is direct and overshoot releases to the volume maximum', () => {
  const {demo} = fixture();
  assert.ok(Math.abs(demo.pureModel(5.125).progress - 0.32) < 1e-10);
  assert.equal(demo.pureModel(6.75).volume, 1.27);
  assert.equal(demo.pureModel(7).volume, 1.27);
  assert.ok(demo.pureModel(7.1).volume < 1.27 && demo.pureModel(7.1).volume > 1);
});

test('every subframe remains finite and within a readable square camera', () => {
  const {demo} = fixture();
  for (let t = 0; t < 14; t += 1 / 240) {
    const m = demo.pureModel(t);
    assert.ok(m.shape.width > 0 && m.shape.height > 0);
    assert.ok(Number.isFinite(m.zoom) && m.zoom > 0);
    assert.ok(m.shape.width * m.zoom < 1300);
    assert.ok(m.shape.height * m.zoom < 1000);
    assert.ok(Number.isFinite(m.cursor.x) && Number.isFinite(m.cursor.y));
  }
});
