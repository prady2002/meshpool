package natprobe

import "testing"

func TestDetectMeasuredNATClasses(t *testing.T) {
	local := Endpoint{IP: "10.0.0.2", Port: 4000}
	pub := Endpoint{IP: "203.0.113.10", Port: 5000}

	tests := []struct {
		name string
		obs  Observation
		want Class
	}{
		{"public", Observation{LocalEndpoint: pub, MappedEndpoints: []Endpoint{pub}, UDPReachable: true}, Public},
		{"full cone", Observation{LocalEndpoint: local, MappedEndpoints: []Endpoint{{IP: "198.51.100.1", Port: 62000}}, UDPReachable: true, ReceivedFromUnknownAddr: true}, FullCone},
		{"address restricted", Observation{LocalEndpoint: local, MappedEndpoints: []Endpoint{{IP: "198.51.100.1", Port: 62000}}, UDPReachable: true, ReceivedFromKnownAddrOtherPort: true}, AddressRestricted},
		{"port restricted", Observation{LocalEndpoint: local, MappedEndpoints: []Endpoint{{IP: "198.51.100.1", Port: 62000}}, UDPReachable: true}, PortRestricted},
		{"symmetric", Observation{LocalEndpoint: local, MappedEndpoints: []Endpoint{{IP: "198.51.100.1", Port: 62000}, {IP: "198.51.100.1", Port: 62001}}, UDPReachable: true}, Symmetric},
		{"cgnat", Observation{LocalEndpoint: local, UDPReachable: false}, CGNAT},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := Detect(tc.obs); got != tc.want {
				t.Fatalf("Detect()=%s want %s", got, tc.want)
			}
		})
	}
}

func TestCanHolePunchPhysics(t *testing.T) {
	if CanHolePunch(Symmetric, Symmetric) {
		t.Fatal("symmetric-symmetric hole punching must remain structural 0%")
	}
	if !CanHolePunch(FullCone, Symmetric) {
		t.Fatal("full-cone to symmetric should be physically plausible")
	}
	if CanHolePunch(CGNAT, PortRestricted) {
		t.Fatal("CGNAT to residential NAT must relay")
	}
	if KeepAliveSeconds < 20 || KeepAliveSeconds > 25 {
		t.Fatalf("keepalive=%ds, want 20-25s", KeepAliveSeconds)
	}
}
