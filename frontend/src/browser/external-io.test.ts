import { describe, expect, it } from 'vitest';
import { encodeWords, sumShares } from './external-io';
import { makeMask, VideoCounter } from './geometry';

describe('browser ExternalIO exact arithmetic', () => {
  it('matches little-endian Z2(64), including carry and wrap beyond JS safe integers', () => {
    const top = (1n << 64n) - 1n;
    const messages = [encodeWords([top, 1n << 63n, top]), encodeWords([2n, 1n << 63n, 1n]), encodeWords([3n, 255n, 0n])];
    expect([...sumShares(messages, 3)]).toEqual([4n, 255n, 0n]);
    expect([...new Uint8Array(encodeWords([-1n]))]).toEqual(Array(8).fill(255));
  });
  it('rejects missing parties and active/wrong-size payloads', () => {
    expect(() => sumShares([encodeWords([1n])], 1)).toThrow();
    expect(() => sumShares([encodeWords([1n]), encodeWords([1n]), encodeWords([1n, 2n])], 1)).toThrow();
  });
});
it('rounds masks outwards, clips and unions overlapping detections', () => {
  expect([...makeMask([[-1, -.5, 2.1, 1.6], [.5, .5, 1, 1]], 2, 2, 2, 2)]).toEqual([1, 1, 1, 1]);
});
it('retains confirmed tracking across chunks and uses the same TTL', () => {
  const counter = new VideoCounter();
  for (let i = 0; i < 3; i++) counter.update([{ box: [10 + i, 10, 20, 20], score: .9, kind: 'person' }], i);
  expect(counter.summary().people).toBe(1);
  for (let i = 3; i < 65; i++) counter.update([], i);
  for (let i = 65; i < 68; i++) counter.update([{ box: [10, 10, 20, 20], score: .9, kind: 'person' }], i);
  expect(counter.summary().people).toBe(2);
});
