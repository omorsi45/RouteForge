package router

// Decision reasons added by the quality engine. ReasonPinned and
// ReasonStrongest are declared with the decision record.
const (
	// ReasonCheapestPassing: the cheapest eligible model whose lcb reached the
	// profile's lambda.
	ReasonCheapestPassing = "cheapest_passing"
	// ReasonEscalated: no eligible model reached lambda, so the policy's
	// strongest model (or the best eligible one) was chosen.
	ReasonEscalated = "escalated_no_model_passed"
	// ReasonCheapest: the cheapest strategy's pick, by expected cost.
	ReasonCheapest = "cheapest"
	// ReasonStaticFamily: the static strategy's model for the prompt's family.
	ReasonStaticFamily = "static_family"
	// ReasonLatency: the latency strategy's pick, by predicted p95.
	ReasonLatency = "lowest_latency"
	// ReasonPickIneligible: a strategy's own pick was ineligible, so the
	// eligible candidate with the highest p was chosen (Ruling R4).
	ReasonPickIneligible = "strategy_pick_ineligible"
)

// Reasons a candidate model is ineligible. When several apply they are listed
// in this order.
const (
	IneligibleBreakerOpen   = "breaker_open"
	IneligibleQueueFull     = "queue_full"
	IneligibleContextWindow = "context_window"
	IneligibleLatencySLO    = "latency_slo"
	IneligibleBudget        = "budget"
)
