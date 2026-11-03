# Run with the project's official Julia 1.4.1 binary.
# This reproduces model_comparison.jl:kfold_splits without fitting any model.
using Random: MersenneTwister, randperm

VERSION == v"1.4.1" || error("Use .tools/julia-1.4.1/bin/julia for the archived protocol")
n, k, seed = 25, 5, 123
x = randperm(MersenneTwister(seed), n)
folds = [(train=setdiff(1:n, x[i:k:n]), test=x[i:k:n]) for i in 1:k]
@assert sort(vcat([f.test for f in folds]...)) == collect(1:n)
@assert all(length(f.train) == 20 && length(f.test) == 5 && isempty(intersect(f.train, f.test)) for f in folds)

output = joinpath(@__DIR__, "..", "data", "mouselab_callaway2022_official", "folds.json")
json_array(a) = "[" * join(a, ", ") * "]"
open(output, "w") do io
    println(io, "{")
    println(io, "  \"source\": \"Callaway et al. 2022, official model_comparison.jl:kfold_splits\",")
    println(io, "  \"julia_version\": \"", VERSION, "\",")
    println(io, "  \"rng\": \"Random.MersenneTwister\",")
    println(io, "  \"seed\": ", seed, ",")
    println(io, "  \"trial_index_base\": 1,")
    println(io, "  \"trial_index_meaning\": \"position in each participant's trials.json array\",")
    println(io, "  \"applies_to\": \"all 95 participants, same trial-position folds\",")
    println(io, "  \"permutation\": ", json_array(x), ",")
    println(io, "  \"folds\": [")
    for (i, f) in enumerate(folds)
        print(io, "    {\"fold\": ", i, ", \"official_file_id\": ", f.test[1],
              ", \"train\": ", json_array(f.train), ", \"test\": ", json_array(f.test), "}")
        println(io, i == k ? "" : ",")
    end
    println(io, "  ]\n}")
end
println(output)
for (i, f) in enumerate(folds)
    println("fold ", i, " test=", f.test)
end
