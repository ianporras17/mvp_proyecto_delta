/* Keep weights/operators byte-identical; expose the native dynamic spatial axes
 * that OpenCV DNN already accepts. Generated files are never source models. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const { onnx } = require(path.join(root, 'node_modules/onnxruntime-web/lib/onnxjs/ort-schema/protobuf/onnx.js'));
const out = path.join(root, 'public/models');
fs.mkdirSync(out, { recursive: true });
const models = [
  ['face', 'preprocessing/face_detection_yunet_2023mar.onnx'],
  ['plate', 'detection/models/license_plate_detection_lpd_yunet_2023mar.onnx'],
  ['objects', 'detection/models/object_detection_yolox_2022nov.onnx'],
];
const manifest = {};
for (const [name, relative] of models) {
  const original = fs.readFileSync(path.join(root, '../src/client', relative));
  let generated = original;
  if (name !== 'objects') {
    const model = onnx.ModelProto.decode(original);
    const graph = model.graph;
    const weights = graph.initializer.map(t => Buffer.from(onnx.TensorProto.encode(t).finish()));
    for (const axis of [2, 3]) graph.input[0].type.tensorType.shape.dim[axis] = onnx.TensorShapeProto.Dimension.create({ dimParam: axis === 2 ? 'height' : 'width' });
    for (const output of graph.output) {
      const dims = output.type.tensorType.shape.dim;
      const axis = dims.length === 3 ? 1 : 0;
      dims[axis] = onnx.TensorShapeProto.Dimension.create({ dimParam: `${output.name}_anchors` });
    }
    graph.valueInfo = []; // discard stale shape hints, preserve every operator
    generated = Buffer.from(onnx.ModelProto.encode(model).finish());
    const roundtrip = onnx.ModelProto.decode(generated);
    if (roundtrip.graph.initializer.some((t, i) => !weights[i].equals(Buffer.from(onnx.TensorProto.encode(t).finish())))) throw Error('Weights changed');
  }
  fs.writeFileSync(path.join(out, `${name}.onnx`), generated);
  manifest[name] = { source: relative, source_sha256: crypto.createHash('sha256').update(original).digest('hex'), sha256: crypto.createHash('sha256').update(generated).digest('hex'), bytes: generated.length };
}
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2));
for (const [from, to] of [['preprocessing/LICENSE', 'LICENSE-YuNet'], ['detection/models/LICENSE', 'LICENSE-models']]) fs.copyFileSync(path.join(root, '../src/client', from), path.join(out, to));
for (const [pkg, files] of [['onnxruntime-web', ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']], ['@ffmpeg/core', ['ffmpeg-core.js', 'ffmpeg-core.wasm']]]) {
  const destination = path.join(root, 'public/vendor', pkg === 'onnxruntime-web' ? 'ort' : 'ffmpeg');
  fs.mkdirSync(destination, { recursive: true });
  for (const file of files) fs.copyFileSync(path.join(root, 'node_modules', pkg, 'dist', pkg === 'onnxruntime-web' ? '' : 'esm', file), path.join(destination, file));
}
console.log('Local ONNX models, licenses and WASM runtimes prepared.');
