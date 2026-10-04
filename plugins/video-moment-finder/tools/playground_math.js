/* Real 2D Euclidean dot-product model. Shared by the offline UI and Node checks. */
function vectorMetrics(v, w) {
  if (![v, w].every(a => Array.isArray(a) && a.length === 2 && a.every(Number.isFinite))) {
    throw new Error('Two finite 2D vectors are required');
  }
  const dot = v[0] * w[0] + v[1] * w[1];
  const nv = Math.hypot(...v), nw = Math.hypot(...w);
  const cosine = nv && nw ? Math.max(-1, Math.min(1, dot / nv / nw)) : null;
  const scalar = nv ? dot / nv : null;
  const projection = nv ? v.map(x => x / nv * scalar) : null;
  return {dot, nv, nw, cosine, scalar, projection,
    // atan2 stays accurate near parallel/opposing vectors, where acos loses precision.
    angle: cosine === null ? null : Math.atan2(Math.abs(v[0] * w[1] - v[1] * w[0]), dot) * 180 / Math.PI,
    components: [v[0] * w[0], v[1] * w[1]]};
}
if (typeof module !== 'undefined') module.exports = {vectorMetrics};
