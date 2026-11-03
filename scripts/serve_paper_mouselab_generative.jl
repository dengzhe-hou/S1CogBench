# JSONL interface to the original Callaway Exp1 rollout, for external policies.
using JSON
using Random

VERSION == v"1.4.1" || error("Use the project-local Julia 1.4.1 binary")
length(ARGS) == 1 || error("Pass the compatible planning/model directory")
const MODEL_DIRECTORY = abspath(first(ARGS))
cd(MODEL_DIRECTORY)
empty!(ARGS)
push!(ARGS, "exp1")
# Original includes may print diagnostics; stdout is reserved for JSONL.
redirect_stdout(stderr) do
    include(joinpath(MODEL_DIRECTORY, "base.jl"))
    include(joinpath(MODEL_DIRECTORY, "space.jl"))
    include(joinpath(MODEL_DIRECTORY, "abstract_model.jl"))
    include(joinpath(MODEL_DIRECTORY, "simulation.jl"))
end

const ORIGINAL_TRIALS = load_trials(EXPERIMENT) |> OrderedDict |> sort!
const PARTICIPANTS = collect(keys(ORIGINAL_TRIALS))
@assert length(PARTICIPANTS) == 95
@assert all(length(ts) == 25 for ts in values(ORIGINAL_TRIALS))
@assert all(t.m.cost == 0 && t.m.expand_only && id(t.m) == "4O8nSiCFkbB"
            for ts in values(ORIGINAL_TRIALS) for t in ts)

visible_belief(b) = [isnan(value) ? nothing : value for value in b]

function ready_message()
    m = first(values(ORIGINAL_TRIALS))[1].m
    Dict("type" => "ready", "participants" => PARTICIPANTS,
        "trials_per_participant" => 25, "repeats" => 10,
        "expected_episodes" => 23750, "environment_cost" => m.cost,
        "prior" => Dict("values" => support(m.rewards[2]),
                        "probabilities" => probs(m.rewards[2]),
                        "independent_nodes" => true, "known_root_value" => 0),
        "graph" => [[child - 1 for child in children] for children in m.graph],
        "mapping" => Dict("text_nodes" => "zero-based; root 0",
            "official_nodes" => "Julia one-based; root 1; TERM 0",
            "action_keys" => "STOP -> TERM=0; Rk -> Julia node k+1",
            "official_best_path" => "Julia one-based, excluding root",
            "best_path" => "zero-based, excluding root"),
        "metadata" => Dict("task" => "Callaway 2022 Exp1 official meta-MDP",
            "mdp_id" => id(m), "expand_only" => m.expand_only,
            "source_model_directory" => MODEL_DIRECTORY,
            "dynamics_source" => joinpath(MODEL_DIRECTORY, "mdp.jl"),
            "schedule_source" => joinpath(MODEL_DIRECTORY, "simulate.jl"),
            "stop_semantics" => "Original rollout and term_reward: best-path expected value; no model route response required.",
            "original_seed_status" => "Original simulate.jl does not set a simulation seed; CV seed 123 is not a simulation seed.",
            "replay_adaptation" => "New explicit MersenneTwister seed pre-samples independent hidden node values from original rewards. Original rollout(pol,s) and observe!(m,b,s,c) execute them. Equal-distribution paired replay; not author or human hidden trajectories.",
            "input_boundary" => "Only observed values, history and legal actions before STOP; hidden ground_truth only in terminal scoring output.",
            "human_wait" => "Human 3-second reveal wait is not simulated or converted to point cost; environment cost stays zero."))
end

mutable struct ExternalPolicy <: Policy
    m::MetaMDP
    metadata::Dict{String,Any}
    history::Vector{Tuple{Int,Float64}}
    requests::Channel{Any}
    actions::Channel{Int}
    last_action::Int
    decision::Int
end

function (pol::ExternalPolicy)(b::Belief)
    if pol.last_action != TERM
        push!(pol.history, (pol.last_action - 1, b[pol.last_action]))
    end
    pol.decision += 1
    official_actions = [TERM; [c for c in eachindex(b) if allowed(pol.m, b, c)]]
    legal_actions = [c == TERM ? "STOP" : "R$(c-1)" for c in official_actions]
    metadata = pol.metadata
    request_id = "mouselab-meta/$(metadata["participant"])/$(metadata["trial"])/$(metadata["repeat"])/$(pol.decision)"
    put!(pol.requests, Dict("type" => "action_request", "id" => request_id,
        "metadata" => metadata, "done" => false,
        "history" => [Any[node, value] for (node, value) in pol.history],
        "belief" => visible_belief(b),
        "observed_values" => [Any[i-1, b[i]] for i in eachindex(b) if !isnan(b[i])],
        "legal_actions" => legal_actions, "official_actions" => official_actions,
        "reveals" => length(pol.history)))
    action = take!(pol.actions)
    action in official_actions || error("Illegal original Mouselab action $action")
    pol.last_action = action
    action
end

mutable struct EpisodeSession
    policy::ExternalPolicy
    task::Task
    last_message::Dict{String,Any}
    done::Bool
end

function start_episode(command)
    participant = String(command["participant"])
    haskey(ORIGINAL_TRIALS, participant) || error("Unknown original participant $participant")
    trial = Int(command["trial"])
    repetition = Int(command["repeat"])
    seed = Int(command["seed"])
    1 <= trial <= 25 || error("trial must be in 1:25")
    1 <= repetition <= 10 || error("repeat must be in 1:10")
    participant_index = findfirst(==(participant), PARTICIPANTS)
    episode = (participant_index - 1) * 250 + (repetition - 1) * 25 + trial
    metadata = Dict{String,Any}("episode" => episode, "participant" => participant,
        "trial" => trial, "repeat" => repetition, "seed" => seed)
    m = ORIGINAL_TRIALS[participant][trial].m
    rng = MersenneTwister(seed)
    # Root is known 0; all other nodes retain their original independent priors.
    truth = [0.; [Float64(rand(rng, m.rewards[c])) for c in 2:length(m)]]
    requests = Channel{Any}(1)
    actions = Channel{Int}(1)
    policy = ExternalPolicy(m, metadata, Tuple{Int,Float64}[], requests, actions, TERM, 0)
    task = @async begin
        result = rollout(policy, truth)
        @assert result.reward == term_reward(m, result.belief)
        @assert result.n_steps - 1 == length(policy.history)
        path = best_path(m, result.belief)
        put!(requests, Dict("type" => "done", "metadata" => metadata, "done" => true,
            "history" => [Any[node, value] for (node, value) in policy.history],
            "terminal_expected_reward" => result.reward,
            "reveals" => result.n_steps - 1,
            "official_best_path" => path, "best_path" => path .- 1,
            "terminal_belief" => visible_belief(result.belief),
            "ground_truth" => truth))
    end
    # Propagate real rollout failures to the waiting client instead of hanging.
    bind(requests, task)
    bind(actions, task)
    message = take!(requests)
    EpisodeSession(policy, task, message, message["done"])
end

function step_episode(session, command)
    session.done && error("Episode already stopped; reset the next original trial")
    action_key = String(command["action"])
    index = findfirst(==(action_key), session.last_message["legal_actions"])
    index === nothing && error("Illegal action $action_key; choose one of $(session.last_message["legal_actions"])")
    action = session.last_message["official_actions"][index]
    put!(session.policy.actions, action)
    message = take!(session.policy.requests)
    session.last_message = message
    session.done = message["done"]
    message
end

function emit_json(value)
    JSON.print(stdout, value)
    println(stdout)
    flush(stdout)
end

function serve()
    emit_json(ready_message())
    session = nothing
    for line in eachline(stdin)
        command = JSON.parse(line)
        op = command["op"]
        if op == "reset"
            session !== nothing && !session.done && error("Finish the active original rollout before reset")
            session = start_episode(command)
            emit_json(session.last_message)
        elseif op == "step"
            session === nothing && error("reset must start an episode before step")
            emit_json(step_episode(session, command))
        elseif op == "quit"
            emit_json(Dict("type" => "bye"))
            return
        else
            error("Unknown operation $op; use reset, step or quit")
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    serve()
end
