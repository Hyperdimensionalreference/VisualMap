'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const occupancy = require('../../src/visualmap/visualization/viewer_occupancy.js');
const box = (min, max, world_id = 'world-1') => ({min, max, world_id});
const sub = (a, b) => a.map((value, index) => value - b[index]);
const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = point => point.map(value => value / Math.hypot(...point));

function camera(eye = [0, 0, 0], target = [0, 0, 1]) {
  const z = unit(sub(target, eye)), x = unit(cross(z, [0, -1, 0]));
  const basis = {eye, z, x, y: unit(cross(x, z))};
  const width = 400, height = 300, near = .1;
  const project = point => {
    const relative = sub(point, eye), depth = dot(relative, z);
    if (depth <= near) return null;
    return {x: width / 2 + dot(relative, x) * 180 / depth,
      y: height / 2 - dot(relative, basis.y) * 180 / depth, z: depth};
  };
  return {basis, project, width, height, near};
}

function drawing() {
  const calls = [];
  const context = {calls, save() {}, restore() {}, globalAlpha: 1};
  for (const method of ['beginPath', 'moveTo', 'lineTo', 'closePath', 'setLineDash', 'stroke', 'fill']) {
    context[method] = (...args) => calls.push({method, args, fillStyle: context.fillStyle,
      strokeStyle: context.strokeStyle, globalAlpha: context.globalAlpha, lineWidth: context.lineWidth});
  }
  return context;
}

test('module exposes the same browser API without a network or DOM dependency', () => {
  const source = fs.readFileSync(path.join(__dirname, '../../src/visualmap/visualization/viewer_occupancy.js'), 'utf8');
  const sandbox = {window: {}};
  vm.runInNewContext(source, sandbox);
  assert.deepEqual(Object.keys(sandbox.window.VisualMapOccupancy), Object.keys(occupancy));
  assert.equal(Object.isFrozen(sandbox.window.VisualMapOccupancy), true);
});

test('cuboid has eight distinct extent corners and never substitutes missing dimensions', () => {
  const bounds = box([-2, 3, 5], [7, 11, 13]);
  const corners = occupancy.corners(bounds);
  assert.equal(corners.length, 8);
  assert.equal(new Set(corners.map(JSON.stringify)).size, 8);
  for (const point of corners) point.forEach((value, index) => assert.ok([bounds.min[index], bounds.max[index]].includes(value)));
  for (const invalid of [null, box([0, 0, 0], [1, 0, 1]), box([0, 0, 0], [1, Infinity, 1]),
    box([0, 0, 0], [-1, 1, 1]), {...bounds, valid: false}]) {
    assert.deepEqual(occupancy.corners(invalid), []);
  }
});

test('boxes from unknown or different coordinate worlds cannot imply contact', () => {
  const a = box([0, 0, 0], [2, 2, 2]);
  const b = box([1, 1, 1], [3, 3, 3]);
  assert.equal(occupancy.potentialOverlap(a, {...b, world_id: undefined}).reason, 'missing_world');
  assert.equal(occupancy.potentialOverlap(a, {...b, world_id: ''}).reason, 'missing_world');
  assert.equal(occupancy.potentialOverlap(a, {...b, world_id: 'world-2'}).reason, 'different_worlds');
  assert.equal(occupancy.potentialOverlap(a, {...b, min: [NaN, 1, 1]}).reason, 'invalid_bounds');
});

test('separation reports the shortest diagonal gap in uncalibrated map units', () => {
  const result = occupancy.potentialOverlap(box([0, 0, 0], [1, 1, 1]), box([2, 3, .5], [3, 4, 1.5]));
  assert.equal(result.usable, true);
  assert.equal(result.overlap, false);
  assert.deepEqual(result.axis_gaps, [1, 2, 0]);
  assert.equal(result.gap, Math.sqrt(5));
  assert.equal(result.units, 'map_units');
  assert.equal(result.penetration_axis, null);
});

test('touching faces and edges are distinct from a positive overlap volume', () => {
  const a = box([0, 0, 0], [1, 1, 1]);
  for (const b of [box([1, .2, .2], [2, .8, .8]), box([1, 1, .2], [2, 2, .8])]) {
    const result = occupancy.potentialOverlap(a, b);
    assert.equal(result.touching, true);
    assert.equal(result.overlap, false);
    assert.equal(result.gap, 0);
    assert.equal(result.penetration, 0);
  }
});

test('overlap reports the minimum separating axis displacement', () => {
  const result = occupancy.potentialOverlap(box([0, 0, 0], [2, 2, 2]), box([1.5, 1, 1], [3, 4, 4]));
  assert.equal(result.overlap, true);
  assert.equal(result.penetration, .5);
  assert.equal(result.penetration_axis, 'x');
  assert.deepEqual(result.axis_penetration, [.5, 1, 1]);
});

test('contained box uses exit displacement rather than calling its width penetration', () => {
  const result = occupancy.potentialOverlap(box([0, 0, 0], [10, 10, 10]), box([4, 4, 4], [6, 6, 6]));
  assert.deepEqual(result.intersection_extents, [2, 2, 2]);
  assert.deepEqual(result.axis_penetration, [6, 6, 6]);
  assert.equal(result.penetration, 6);
});

test('safety margin expands each object separately and retains the original gap', () => {
  const a = box([0, 0, 0], [1, 1, 1]), b = box([1.5, 0, 0], [2.5, 1, 1]);
  const result = occupancy.potentialOverlap(a, b, {margin: .3});
  assert.equal(result.raw_overlap, false);
  assert.equal(result.overlap, true);
  assert.equal(result.gap, .5);
  assert.equal(result.buffered_gap, 0);
  assert.ok(Math.abs(result.penetration - .1) < 1e-9);
  assert.equal(result.margin_per_object, .3);
  for (const margin of [-1, NaN, Infinity]) assert.equal(occupancy.potentialOverlap(a, b, {margin}).reason, 'invalid_margin');
});

test('overlap calculations and expanded envelopes preserve caller geometry', () => {
  const a = Object.freeze({min: Object.freeze([0, 0, 0]), max: Object.freeze([1, 1, 1]), world_id: 'w'});
  const b = Object.freeze({min: Object.freeze([.5, .5, .5]), max: Object.freeze([2, 2, 2]), world_id: 'w'});
  const before = JSON.stringify([a, b]);
  occupancy.potentialOverlap(a, b, {margin: .2});
  const expanded = occupancy.expandBounds(a, .2);
  expanded.min[0] = -10;
  assert.equal(JSON.stringify([a, b]), before);
});

test('an oblique camera reveals three filled faces with stable far-to-near depth ordering', () => {
  const bounds = box([-1, -1, 3], [1, 1, 5]);
  const options = camera([6, -4, -7], [0, 0, 4]);
  const faces = occupancy.prepareFaces([{trackId: 1, bounds}], options);
  assert.deepEqual(new Set(faces.map(face => face.faceIndex)), new Set([1, 2, 4]));
  assert.equal(faces.length, 3);
  assert.ok(faces.every((face, index) => index === 0 || faces[index - 1].depth >= face.depth));
  assert.ok(faces.every(face => face.polygon.length >= 3 && face.polygon.every(p => Number.isFinite(p.x) && Number.isFinite(p.y))));
});

test('a volume crossing the camera near plane keeps its visible portion', () => {
  const options = camera();
  const bounds = box([.5, -.5, -.5], [2, .5, 2]);
  const faces = occupancy.prepareFaces([{trackId: 3, bounds}], options);
  assert.ok(faces.length > 0, 'partly visible object must not disappear because some corners are behind the eye');
  for (const face of faces) {
    assert.ok(face.depth > options.near);
    assert.ok(face.polygon.every(p => p.x >= -1e-8 && p.x <= options.width + 1e-8 && p.y >= -1e-8 && p.y <= options.height + 1e-8));
  }
  const clipped = occupancy.clipNear([[-1, 0, -.5], [1, 0, -.5], [1, 1, 1], [-1, 1, 1]], options.basis, options.near);
  assert.equal(clipped.length, 4);
  assert.ok(clipped.every(p => p[2] > options.near));
});

test('orthographic faces follow parallel rays rather than the eye-to-object offset', () => {
  const options = camera([20, 20, -100], [21, 21, -99]), center = [0, 0, 4];
  options.orthographic = true;
  options.project = point => ({x: 200 + dot(sub(point, center), options.basis.x) * 40,
    y: 150 - dot(sub(point, center), options.basis.y) * 40,
    z: dot(sub(point, options.basis.eye), options.basis.z)});
  const faces = occupancy.prepareFaces([{trackId: 1, bounds: box([-1, -1, 3], [1, 1, 5])}], options);
  assert.deepEqual(new Set(faces.map(face => face.faceIndex)), new Set([0, 2, 4]));
});

test('a camera inside an envelope sees clipped surfaces while entirely behind geometry is hidden', () => {
  const options = camera();
  assert.ok(occupancy.prepareFaces([{trackId: 1, bounds: box([-1, -1, -1], [1, 1, 2])}], options).length > 0);
  assert.deepEqual(occupancy.prepareFaces([{trackId: 1, bounds: box([-1, -1, -3], [1, 1, -1])}], options), []);
});

test('viewport clipping retains a face crossing all four edges without nonfinite vertices', () => {
  const polygon = occupancy.clipViewport([{x: -100, y: -100, z: 1}, {x: 500, y: -100, z: 1},
    {x: 500, y: 400, z: 1}, {x: -100, y: 400, z: 1}], 400, 300);
  assert.equal(polygon.length, 4);
  assert.deepEqual(new Set(polygon.map(p => `${p.x},${p.y}`)), new Set(['0,0', '400,0', '400,300', '0,300']));
});

test('batch drawing paints solid gray faces and selects the frontmost volume rather than its center', () => {
  const context = drawing();
  const result = occupancy.render(context, [{trackId: 'near', bounds: box([-1, -1, 2], [1, 1, 3])},
    {trackId: 'far', bounds: box([-1, -1, 4], [1, 1, 6])}], {...camera(), selectedTrackId: 'near'});
  const fills = context.calls.filter(call => call.method === 'fill');
  assert.equal(fills.length, result.faces.length);
  assert.ok(fills.length >= 2);
  assert.ok(fills.every(call => /^#[a-f\d]{6}$/i.test(call.fillStyle) && call.globalAlpha >= .5 && call.globalAlpha <= .65));
  assert.equal(occupancy.pickAt(240, 190, result.hitTargets).trackId, 'near');
  assert.equal(occupancy.pickAt(10, 10, result.hitTargets), null);
  assert.ok(context.calls.some(call => call.method === 'stroke' && call.lineWidth === 2.2));
  assert.equal(context.globalAlpha, 1);
});

test('observed and inferred envelopes retain the same solid faces with different edge evidence', () => {
  const context = drawing();
  occupancy.render(context, [{trackId: 1, bounds: box([-1, -1, 2], [1, 1, 3]), extent_kind: 'observed'},
    {trackId: 2, bounds: box([-1, -1, 4], [1, 1, 5]), extent_kind: 'inferred'}], camera());
  const dashes = context.calls.filter(call => call.method === 'setLineDash').map(call => call.args[0]);
  assert.ok(dashes.some(dash => dash.length === 0));
  assert.ok(dashes.some(dash => dash.join(',') === '5,4'));
  assert.equal(context.calls.filter(call => call.method === 'fill').length, 2);
});

test('polygon hit testing includes its boundary and excludes concave cutouts', () => {
  const polygon = [{x: 0, y: 0}, {x: 10, y: 0}, {x: 10, y: 4}, {x: 4, y: 4}, {x: 4, y: 10}, {x: 0, y: 10}];
  assert.equal(occupancy.pointInPolygon(0, 5, polygon), true);
  assert.equal(occupancy.pointInPolygon(2, 8, polygon), true);
  assert.equal(occupancy.pointInPolygon(8, 8, polygon), false);
  assert.equal(occupancy.pointInPolygon(NaN, 5, polygon), false);
});

test('multi-view convex proxy draws its carved faces rather than the larger AABB', () => {
  const bounds = {...box([-1, -1, 3], [1, 1, 5]),
    shape_vertices: [[-1,-1,3],[1,-1,3],[0,1,3],[0,0,5]],
    shape_faces: [[0,2,1],[0,1,3],[1,2,3],[2,0,3]]};
  const faces = occupancy.prepareFaces([{trackId: 4, bounds}], {...camera(), backfaceCull: false});
  assert.equal(faces.length, 4);
  assert.ok(faces.every(face => face.polygon.length === 3));
  assert.ok(faces.every(face => face.polygon.every(point => Number.isFinite(point.x) && Number.isFinite(point.y))));
  const rotated = occupancy.prepareFaces([{trackId: 4, bounds}], camera([6,-4,-7],[0,0,4]));
  assert.ok(rotated.length > 0 && rotated.length < 4);
});

test('invalid shape indices fall back to the compatible envelope without throwing', () => {
  const bounds = {...box([-1,-1,3],[1,1,5]), shape_vertices: [[0,0,4]], shape_faces: [[0,1,2]]};
  assert.ok(occupancy.prepareFaces([{trackId: 5, bounds}], camera()).length > 0);
});

test('eye in the loose AABB but outside the carved solid still culls its back faces', () => {
  const bounds = {...box([-1,-1,3],[1,1,5]),
    shape_vertices: [[-1,-1,3],[1,-1,3],[0,1,3],[0,0,5]],
    shape_faces: [[0,2,1],[0,1,3],[1,2,3],[2,0,3]]};
  const options = camera([.8,.8,3.1],[.8,.8,4.1]);
  const faces = occupancy.prepareFaces([{trackId: 4, bounds}], options);
  assert.deepEqual(faces.map(face => face.faceIndex), [2]);
});
