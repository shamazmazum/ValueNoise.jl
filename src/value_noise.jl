function lolrng(x    :: Integer,
                y    :: Integer,
                z    :: Integer,
                seed :: Integer)
    r1 = UInt32(x) * UInt32(0x1B873593)
    r2 = UInt32(y) * UInt32(0x19088711)
    r3 = UInt32(z) * UInt32(0xB2D05E13)

    r  = UInt32(seed) + r1 + r2 + r3
    r ⊻= r >> UInt32(5)
    r *= UInt32(0xCC9E2D51)
    return r / 0xffffffff
end

function octave(x    :: Real,
                y    :: Real,
                z    :: Real,
                oct  :: Integer,
                seed :: Integer)
    divisor = 2.0^(-oct)

    xi = (x / divisor) |> floor |> Int
    yi = (y / divisor) |> floor |> Int
    zi = (z / divisor) |> floor |> Int

    δx = rem(x, divisor) / divisor
    δy = rem(y, divisor) / divisor
    δz = rem(z, divisor) / divisor

    v000 = lolrng(xi,     yi,     zi, seed)
    v001 = lolrng(xi + 1, yi,     zi, seed)
    v010 = lolrng(xi,     yi + 1, zi, seed)
    v011 = lolrng(xi + 1, yi + 1, zi, seed)

    v100 = lolrng(xi,     yi,     zi + 1, seed)
    v101 = lolrng(xi + 1, yi,     zi + 1, seed)
    v110 = lolrng(xi,     yi + 1, zi + 1, seed)
    v111 = lolrng(xi + 1, yi + 1, zi + 1, seed)

    inter(v1, v2, x) = v1 + (v2 - v1)*x
    v00 = inter(v000, v001, δx)
    v01 = inter(v010, v011, δx)
    v10 = inter(v100, v101, δx)
    v11 = inter(v110, v111, δx)

    v0 = inter(v00, v01, δy)
    v1 = inter(v10, v11, δy)

    v = inter(v0, v1, δz)
    return v
end

"""
    value_noise(x :: Real, y :: Real, z :: Real, octaves :: Integer, seed :: Integer)

Calculate value noise at the point (x,y,z).

Calaculated noise consists of `octaves` octaves (higher
harmonics). `seed` is a seed for random number generator.

# Examples
```jddoctest
julia> [value_noise(i/10, 0, 0, 10, 1343) for i in 1:20]
20-element Vector{Float64}:
 0.6782011856880998
 0.6811537717432827
 0.6749341884205909
 0.6837415201436138
 0.7459242823636113
 0.6684127212046574
 0.6642874704554677
 0.6833569738117538
 0.7214221918747463
 0.8051134343235813
 0.6986164658038979
 0.647098867616866
 0.6158556023606773
 0.6356918681071072
 0.6617448759515002
 0.6710733979362564
 0.7094478163638382
 0.7446686007256124
 0.7987984329263587
 0.9086695925335629
```
"""
function value_noise(x       :: Real,
                     y       :: Real,
                     z       :: Real,
                     octaves :: Integer,
                     seed    :: Integer)
    mapreduce((x, octave) -> x ./ 2.0^octave, +,
              (octave(x, y, z, o, seed) for o in 0:octaves-1),
              countfrom(0)) ./ 2*(1 - 2.0^(-octaves))
end
