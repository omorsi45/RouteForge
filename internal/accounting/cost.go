package accounting

import (
	"fmt"
	"math"
)

// nanosPerUSD is how many nano-USD make one USD.
const nanosPerUSD = 1e9

// maxUSDNanos is the largest amount, in nano-USD, that USDToNanos converts. A
// float64 holds every whole number up to 2^53, but the product usd*1e9 is
// itself rounded, and from about 2^51 up that rounding can move the result by
// a whole nano-USD. Below this limit the conversion is exact; it is about 1.1
// million USD.
const maxUSDNanos = 1 << 50

// EstimateTokens estimates the tokens in bytes bytes of text: one token per
// four bytes, rounded up.
func EstimateTokens(bytes int) int { return (bytes + 3) / 4 }

// CostNanos is the cost in nano-USD of promptTokens input tokens and
// completionTokens output tokens at price p.
func CostNanos(p Price, promptTokens, completionTokens int) int64 {
	return int64(promptTokens)*p.InNanos + int64(completionTokens)*p.OutNanos
}

// ReservationNanos is what a request holds back before its call: every prompt
// byte priced as a token, which can only overstate the prompt, plus the whole
// output cap at the output price.
func ReservationNanos(p Price, promptBytes, maxOutputTokens int) int64 {
	return int64(promptBytes)*p.InNanos + int64(maxOutputTokens)*p.OutNanos
}

// NanosToUSD converts nano-USD to USD, for display and for the response.
func NanosToUSD(n int64) float64 { return float64(n) / nanosPerUSD }

// USDToNanos converts a USD amount to nano-USD. The amount must be finite, not
// negative and a whole number of nano-USD, that is, written with at most nine
// decimals; anything else is an error rather than a silent rounding.
//
// config validates tenant budgets with the same rule, because it cannot import
// this package (accounting imports config); a test keeps the two in step.
func USDToNanos(usd float64) (int64, error) {
	if math.IsNaN(usd) || math.IsInf(usd, 0) || usd < 0 {
		return 0, fmt.Errorf("USD amount %v must be finite and not negative", usd)
	}
	scaled := math.Round(usd * nanosPerUSD)
	if scaled > maxUSDNanos {
		return 0, fmt.Errorf("USD amount %v is too large to convert to nano-USD exactly", usd)
	}
	n := int64(scaled)
	// The nearest double to n/1e9 is what a decimal with at most nine places
	// parses to, so equality means usd is exactly n nano-USD.
	if NanosToUSD(n) != usd {
		return 0, fmt.Errorf("USD amount %v is not a whole number of nano-USD (at most 9 decimals)", usd)
	}
	return n, nil
}
