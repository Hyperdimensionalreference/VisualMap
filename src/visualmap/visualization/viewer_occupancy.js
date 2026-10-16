(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.VisualMapOccupancy = api;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  const EPS = 1e-9;
  const finitePoint = point => Array.isArray(point) && point.length === 3 && point.every(Number.isFinite);
  const sub = (a, b) => a.map((value, index) => value - b[index]);
  const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
  const mix = (a, b, t) => a.map((value, index) => value + (b[index] - value) * t);
  const faceDefinitions = [
    {indices: [0, 3, 7, 4], normal: [-1, 0, 0], shade: '#b8b7b0'},
    {indices: [1, 5, 6, 2], normal: [1, 0, 0], shade: '#aaa9a2'},
    {indices: [0, 4, 5, 1], normal: [0, -1, 0], shade: '#d3d2cc'},
    {indices: [3, 2, 6, 7], normal: [0, 1, 0], shade: '#aaa9a2'},
    {indices: [0, 1, 2, 3], normal: [0, 0, -1], shade: '#c4c3bc'},
    {indices: [4, 7, 6, 5], normal: [0, 0, 1], shade: '#b2b1aa'},
  ];

  function validBounds(bounds) {
    return Boolean(bounds && bounds.valid !== false && finitePoint(bounds.min) && finitePoint(bounds.max) &&
      bounds.min.every((value, index) => value < bounds.max[index]));
  }

  function corners(bounds) {
    if (!validBounds(bounds)) return [];
    const [x0, y0, z0] = bounds.min, [x1, y1, z1] = bounds.max;
    return [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
  }

  function expandBounds(bounds, margin = 0) {
    if (!validBounds(bounds) || !Number.isFinite(margin) || margin < 0) return null;
    const min = bounds.min.map(value => value - margin), max = bounds.max.map(value => value + margin);
    if (!finitePoint(min) || !finitePoint(max)) return null;
    return {...bounds, min, max};
  }

  function validWorld(world) {
    return typeof world === 'string' && world.trim().length > 0 || Number.isFinite(world);
  }

  // A margin expands each object's envelope. This estimates box intersection,
  // not physical contact: sparse monocular supports do not prove a solid interior.
  function potentialOverlap(a, b, options = {}) {
    const margin = options.margin === undefined ? 0 : options.margin;
    const unavailable = reason => ({usable: false, reason, units: 'map_units'});
    if (!validBounds(a) || !validBounds(b)) return unavailable('invalid_bounds');
    if (!validWorld(a.world_id) || !validWorld(b.world_id)) return unavailable('missing_world');
    if (a.world_id !== b.world_id) return unavailable('different_worlds');
    if (!Number.isFinite(margin) || margin < 0) return unavailable('invalid_margin');
    const expandedA = expandBounds(a, margin), expandedB = expandBounds(b, margin);
    if (!expandedA || !expandedB) return unavailable('invalid_margin');
    const rawDepths = [0, 1, 2].map(index =>
      Math.min(a.max[index], b.max[index]) - Math.max(a.min[index], b.min[index]));
    const depths = [0, 1, 2].map(index => Math.min(expandedA.max[index], expandedB.max[index]) -
      Math.max(expandedA.min[index], expandedB.min[index]));
    const axisGaps = rawDepths.map(value => Math.max(0, -value));
    const bufferedAxisGaps = depths.map(value => Math.max(0, -value));
    const overlap = depths.every(value => value > EPS);
    const touching = !overlap && depths.every(value => value >= -EPS);
    const intersectionExtents = depths.map(value => Math.max(0, value));
    // For containment, intersection width is smaller than the displacement
    // needed to separate the boxes. Return both rather than confuse the two.
    const axisPenetration = overlap ? [0, 1, 2].map(index => Math.min(
      expandedA.max[index] - expandedB.min[index], expandedB.max[index] - expandedA.min[index])) : [0, 0, 0];
    const minDepth = overlap ? Math.min(...axisPenetration) : 0;
    return {
      usable: true, reason: overlap ? 'potential_overlap' : touching ? 'envelopes_touch' : 'separated',
      overlap, touching, raw_overlap: rawDepths.every(value => value > EPS),
      gap: Math.hypot(...axisGaps), axis_gaps: axisGaps,
      buffered_gap: Math.hypot(...bufferedAxisGaps), buffered_axis_gaps: bufferedAxisGaps,
      penetration: minDepth, axis_penetration: axisPenetration, intersection_extents: intersectionExtents,
      penetration_axis: overlap ? ['x', 'y', 'z'][axisPenetration.indexOf(minDepth)] : null,
      margin, margin_per_object: margin, units: 'map_units', world_id: a.world_id,
    };
  }

  function clipPolygon(vertices, signedDistance, interpolate = mix) {
    if (!vertices.length) return [];
    const result = [];
    let previous = vertices[vertices.length - 1], previousDistance = signedDistance(previous);
    for (const vertex of vertices) {
      const distance = signedDistance(vertex), inside = distance >= 0, previousInside = previousDistance >= 0;
      if (inside !== previousInside) {
        const t = previousDistance / (previousDistance - distance);
        result.push(interpolate(previous, vertex, Math.min(1, Math.max(0, t))));
      }
      if (inside) result.push(vertex);
      previous = vertex; previousDistance = distance;
    }
    return result;
  }

  function clipNear(vertices, basis, near) {
    if (!finitePoint(basis?.eye) || !finitePoint(basis?.z) || !Number.isFinite(near) || near <= 0) return [];
    // Stay slightly ahead of the caller's near rejection threshold, avoiding a
    // clipped vertex being rejected again because of floating point rounding.
    const plane = near + Math.max(1e-10, near * 1e-7);
    return clipPolygon(vertices, point => dot(sub(point, basis.eye), basis.z) - plane);
  }

  const interpolateScreen = (a, b, t) => ({x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t});

  function clipViewport(vertices, width, height) {
    if (!(width > 0) || !(height > 0) || !Number.isFinite(width) || !Number.isFinite(height)) return [];
    let polygon = vertices;
    for (const distance of [p => p.x, p => width - p.x, p => p.y, p => height - p.y]) {
      polygon = clipPolygon(polygon, distance, interpolateScreen);
    }
    return polygon;
  }

  function polygonArea(polygon) {
    return Math.abs(polygon.reduce((sum, point, index) => {
      const next = polygon[(index + 1) % polygon.length];
      return sum + point.x * next.y - next.x * point.y;
    }, 0)) / 2;
  }

  function prepareFaces(items, options) {
    const {basis, project, width, height} = options || {}, near = options?.near ?? .001;
    if (!finitePoint(basis?.eye) || !finitePoint(basis?.z) || typeof project !== 'function') return [];
    const faces = [];
    for (const item of items || []) {
      const bounds = item?.bounds;
      let vertices = corners(bounds), definitions = faceDefinitions;
      if (!vertices.length) continue;
      // New results carry a convex shape constrained by multiple views. Keep
      // its AABB for broad overlap tests, but draw the actual bounded proxy.
      if (Array.isArray(bounds.shape_vertices) && bounds.shape_vertices.length >= 4 &&
          bounds.shape_vertices.every(finitePoint) && Array.isArray(bounds.shape_faces) &&
          bounds.shape_faces.length >= 4 && bounds.shape_faces.every(face => Array.isArray(face) &&
            face.length >= 3 && face.every(index => Number.isInteger(index) && index >= 0 && index < bounds.shape_vertices.length))) {
        vertices = bounds.shape_vertices;
        const center = [0, 1, 2].map(axis => vertices.reduce((sum, point) => sum + point[axis], 0) / vertices.length);
        definitions = bounds.shape_faces.map((indices, index) => {
          const a = vertices[indices[0]], u = sub(vertices[indices[1]], a), v = sub(vertices[indices[2]], a);
          let normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
          const length = Math.hypot(...normal);
          normal = length > EPS ? normal.map(value => value / length) : [0, 0, 0];
          if (dot(normal, sub(a, center)) < 0) normal = normal.map(value => -value);
          return {indices, normal, shade: faceDefinitions[index % faceDefinitions.length].shade};
        });
      }
      const eyeInside = definitions.every(definition =>
        dot(definition.normal, sub(basis.eye, vertices[definition.indices[0]])) <= EPS);
      const inferred = (item.extent_kind ?? bounds.evidence_kind ?? 'inferred') !== 'observed';
      definitions.forEach((definition, faceIndex) => {
        const original = definition.indices.map(index => vertices[index]);
        const facing = options?.orthographic ? -dot(definition.normal, basis.z) :
          dot(definition.normal, sub(basis.eye, original[0]));
        if (options?.backfaceCull !== false && (options?.orthographic || !eyeInside) && facing <= EPS) return;
        const clipped = clipNear(original, basis, near);
        if (clipped.length < 3) return;
        const projected = clipped.map(project);
        if (projected.some(point => !point || ![point.x, point.y, point.z].every(Number.isFinite))) return;
        const polygon = clipViewport(projected, width, height);
        if (polygon.length < 3 || polygonArea(polygon) < .1) return;
        faces.push({trackId: item.trackId, faceIndex, polygon,
          depth: clipped.reduce((sum, point) => sum + dot(sub(point, basis.eye), basis.z), 0) / clipped.length,
          selected: item.trackId === options.selectedTrackId, inferred, fill: definition.shade,
          world_id: bounds.world_id, bounds});
      });
    }
    // Canvas uses a painter's approximation. Axis-aligned envelopes may
    // intersect; their pairwise test reports that ambiguity separately.
    return faces.sort((a, b) => b.depth - a.depth || a.faceIndex - b.faceIndex);
  }

  function render(context, items, options) {
    const faces = prepareFaces(items, options), hitTargets = [];
    context.save?.();
    for (const face of faces) {
      context.beginPath();
      context.moveTo(face.polygon[0].x, face.polygon[0].y);
      for (const point of face.polygon.slice(1)) context.lineTo(point.x, point.y);
      if (context.closePath) context.closePath();
      else context.lineTo(face.polygon[0].x, face.polygon[0].y);
      context.globalAlpha = face.inferred ? .62 : .65;
      context.fillStyle = face.fill;
      context.fill();
      context.globalAlpha = 1;
      context.lineWidth = face.selected ? 2.2 : 1;
      context.strokeStyle = face.selected ? '#55524b' : face.inferred ? '#8a877f' : '#706e66';
      context.setLineDash(face.inferred ? [5, 4] : []);
      context.stroke();
      hitTargets.push({trackId: face.trackId, polygon: face.polygon, depth: face.depth});
    }
    context.setLineDash([]);
    context.globalAlpha = 1;
    context.restore?.();
    return {faces, hitTargets, visibleTrackIds: Array.from(new Set(faces.map(face => face.trackId)))};
  }

  function pointInPolygon(x, y, polygon) {
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Array.isArray(polygon) || polygon.length < 3) return false;
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const a = polygon[j], b = polygon[i], cross = (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
      if (Math.abs(cross) < 1e-6 && x >= Math.min(a.x, b.x) - EPS && x <= Math.max(a.x, b.x) + EPS &&
        y >= Math.min(a.y, b.y) - EPS && y <= Math.max(a.y, b.y) + EPS) return true;
      if ((a.y > y) !== (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) inside = !inside;
    }
    return inside;
  }

  function pickAt(x, y, hitTargets) {
    // Prefer the last painted visible face, matching what the user clicked.
    for (let index = (hitTargets?.length || 0) - 1; index >= 0; index--) {
      const hit = hitTargets[index];
      if (pointInPolygon(x, y, hit.polygon)) return hit;
    }
    return null;
  }

  return Object.freeze({validBounds, corners, expandBounds, potentialOverlap, clipNear, clipViewport,
    prepareFaces, render, pointInPolygon, pickAt});
});
