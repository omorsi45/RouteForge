package policy

import "sort"

// Block is one calibration block: raw scores in [Lo, Hi) saw K successes in
// N trials, P = K/N, and LCB is the one-sided Wilson lower bound.
type Block struct {
	Lo, Hi float64
	N, K   int
	P, LCB float64
}

// Calibration maps a raw score to an empirical success rate by block.
type Calibration struct {
	Blocks []Block
}

// newCalibration compiles a validated calibration document.
func newCalibration(d *CalibrationDoc) Calibration {
	blocks := make([]Block, len(d.Blocks))
	for i, b := range d.Blocks {
		blocks[i] = Block{Lo: b.Lo, Hi: b.Hi, N: b.N, K: b.K, P: b.P, LCB: b.LCB}
	}
	return Calibration{Blocks: blocks}
}

// Lookup returns the index of the last block with Lo <= raw. A raw score
// below the first Lo maps to block 0. It is a binary search with no
// interpolation, the counterpart of the trainer's Calibration.lookup.
func (c *Calibration) Lookup(raw float64) int {
	// The first block whose Lo exceeds raw; the block before it is the match.
	i := sort.Search(len(c.Blocks), func(i int) bool { return c.Blocks[i].Lo > raw }) - 1
	return max(i, 0)
}
