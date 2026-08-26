// Package natprobe classifies NAT behaviour from active UDP probe observations.
//
// The operator never configures this by hand: the inputs are what STUN-like
// reflectors actually observed for this socket.  The classifier mirrors the
// Python transport planner so the Go agent can report the same full-cone /
// address-restricted / port-restricted / symmetric / CGNAT vocabulary.
package natprobe

const KeepAliveSeconds = 22

type Class string

const (
	Public            Class = "public"
	FullCone          Class = "full_cone"
	AddressRestricted Class = "address_restricted"
	PortRestricted    Class = "port_restricted"
	Symmetric         Class = "symmetric"
	CGNAT             Class = "cgnat"
)

type Endpoint struct {
	IP   string
	Port int
}

type Observation struct {
	LocalEndpoint                  Endpoint
	MappedEndpoints                []Endpoint
	UDPReachable                   bool
	ReceivedFromUnknownAddr        bool
	ReceivedFromKnownAddrOtherPort bool
	MappedIPIsPrivateOrShared      bool
}

func Detect(obs Observation) Class {
	if !obs.UDPReachable || len(obs.MappedEndpoints) == 0 || obs.MappedIPIsPrivateOrShared {
		return CGNAT
	}
	allLocal := true
	for _, ep := range obs.MappedEndpoints {
		if ep != obs.LocalEndpoint {
			allLocal = false
			break
		}
	}
	if allLocal {
		return Public
	}
	seen := map[Endpoint]bool{}
	for _, ep := range obs.MappedEndpoints {
		seen[ep] = true
	}
	if len(seen) > 1 {
		return Symmetric
	}
	if obs.ReceivedFromUnknownAddr {
		return FullCone
	}
	if obs.ReceivedFromKnownAddrOtherPort {
		return AddressRestricted
	}
	return PortRestricted
}

func CanHolePunch(a, b Class) bool {
	return canOneWay(a, b) && canOneWay(b, a)
}

func canOneWay(a, b Class) bool {
	switch a {
	case Public:
		return true
	case FullCone:
		return b != CGNAT
	case AddressRestricted, PortRestricted:
		return b != Symmetric && b != CGNAT
	case Symmetric:
		return b == Public || b == FullCone
	case CGNAT:
		return b == Public
	default:
		return false
	}
}
