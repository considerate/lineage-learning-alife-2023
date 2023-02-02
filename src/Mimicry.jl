module Mimicry

import UnicodePlots
import Zygote
import PolygonOps
import StaticArrays
import TERMIOS
import Lux
import Optimisers
using Printf
import NamedTupleTools
import Random
using TimerOutputs
import LinearAlgebra
import ColorSchemes

const tau=2*pi
const Polygon = Vector{StaticArrays.SVector{2, Float32}}
const Point = Tuple{Float32, Float32}
const Polar = Tuple{Float32, Float32}
const Bounds = Tuple{Point, Point}
const Arena = Tuple{Polygon, Polygon, Bounds}

const Prob = Tuple{Vector{Float32}, Vector{Float32}}
const Sensors = Matrix{Float32}
const Carry = Tuple{Tuple{Matrix{Float32}, Matrix{Float32}},Tuple{Matrix{Float32}, Matrix{Float32}},Tuple{Matrix{Float32}, Matrix{Float32}}}
const Sampled = Float32
const Parent = Int64

to = TimerOutput()
TimerOutputs.disable_timer!(to)

function cleanup()
    print("\033[?25h") # show cursor
end

struct Network <: Lux.AbstractExplicitLayer
    dense :: Lux.Dense
    dense2 :: Lux.Dense
    means :: Lux.Dense
    logvars :: Lux.Dense
end

#
#  sensor_0   memory_0
#     x    x   x
#    turn_0  (memory_1 (+) memory_0)
#
#  sensor_1  memory_1
#    x  x    x
#    turn_1 (memory_2 (+) memory_1)
#
#  sensor_n  memory_n
#    x  x    x
#    turn_n memory_(n+1)
#
#
# RNN(sensor_(n-21), memory_(n-21), θ) = {turn_(n-20) ... turn_(n+1)}
# L(RNN)

# p(turn_0, ..., turn_n | memory_0, sensor_0, ..., sensor_n, theta)

# p(turn = 'L' | memory, sensor, theta) = sigmoid(...)
#
# (y,m') = W(x,m)j
# loss_0 ~ L(fst(W(x_0, m_0)))
# loss_1 ~ L(fst(W(x_1, m_1))) = L(fst(W(x_1, snd(W(x_0, m_0)))))
# loss_0 + loss_1 = L(fst(W(x_0, m_0))) + L(fst(W(x_1, snd(W(x_0, m_0)))))
#
#
# loss_0 + .... + loss_n
#
#

struct Body
    x::Float32
    y::Float32
    theta::Float32
end

struct CarModel <: Lux.AbstractExplicitContainerLayer{(:lstm_cell, :lstm_cell2, :lstm_cell3, :means, :logvars)}
    lstm_cell:: Lux.LSTMCell
    lstm_cell2:: Lux.LSTMCell
    lstm_cell3:: Lux.LSTMCell
    means :: Lux.Dense
    logvars :: Lux.Dense
end


# A Car decides:
# - what angle to turn using a gaussian
mutable struct Car
    age :: Int64
    lineage :: Int64
    carry :: Carry
    body::Body
    model::CarModel
    parameters::NamedTuple
    state::NamedTuple
    optimiser_state :: NamedTuple
end


# model = CarModel(Lux.LSTMCell(...), Lux.Dense(..), Lux.Dense(..))
# (means, logvars, new_carry) = model((x,carry), ps, st)

# Run one step through the model
function (model::CarModel)(inputs :: Tuple{AbstractMatrix, Carry}, ps :: NamedTuple, st :: NamedTuple)
    (x, (carry, carry2, carry3)) = inputs
    (y, new_carry), st_lstm = model.lstm_cell((x, carry), ps.lstm_cell, st.lstm_cell)
    (y2, new_carry_2), st_lstm = model.lstm_cell((y, carry2), ps.lstm_cell2, st.lstm_cell2)
    (y3, new_carry_3), st_lstm = model.lstm_cell((y2, carry3), ps.lstm_cell3, st.lstm_cell3)
    means, st_means = model.means(y3, ps.means, st.means)
    logvars, st_logvars = model.logvars(y3, ps.logvars, st.logvars)
    st = merge(st, (lstm_cell=st_lstm, means=st_means, logvars=st_logvars))
    return (means, logvars, (new_carry, new_carry_2, new_carry_3)), st
end

function sequence_loss(model :: CarModel, initialcarry :: Carry, sequence :: Vector{Tuple{Matrix{Float32}, Float32}}, ps :: NamedTuple, st :: NamedTuple)
    carry = initialcarry
    loss = 0.0
    for (sensors, sampled) in sequence
        (means, logvars, carry), st = model((sensors, carry), ps, st)
        loss = loss + gaussloss(means, logvars, sampled)
    end
    return loss, st
end

function train(agent, history :: Vector{Tuple{Tuple{Matrix{Float32}, Float32}, Carry}})
    if length(history) == 0
        return
    end
    (_, carry) = history[1]
    sequence = first.(history)
    (_, st), back = Zygote.pullback(p -> sequence_loss(agent.model, carry, sequence, p, agent.state), agent.parameters)
    agent.state = st
    grads = back((1.0, nothing))[1]
    (st_opt, ps) = Optimisers.update!(agent.optimiser_state, agent.parameters, grads)
    agent.optimiser_state = st_opt
    agent.parameters = ps
end
function newcarry(rng, sizes, batchsize=1) :: Carry
    (a,b,c) = sizes
    return (
            (Lux.zeros32(rng, a, batchsize), Lux.zeros32(rng, a, batchsize)),
            (Lux.zeros32(rng, b, batchsize), Lux.zeros32(rng, b, batchsize)),
            (Lux.zeros32(rng, c, batchsize), Lux.zeros32(rng, c, batchsize)),
           )
end

function Car(rng, learning_rate, sensorSize, arena, batchsize=1)
    (a,b,c) = (40,30,20)
    model = CarModel(Lux.LSTMCell(sensorSize => a, use_bias=true),
                     Lux.LSTMCell(a => b, use_bias=true),
                     Lux.LSTMCell(b => c, use_bias=true),
                     Lux.Dense(c => 1),
                     Lux.Dense(c => 1)
                    )
    (ps, st) = Lux.setup(rng, model)
    body = randomBody(arena)
    st_opt = Optimisers.setup(Optimisers.Descent(learning_rate), ps)
    carry = newcarry(rng, (a,b,c), batchsize)
    return Car(0, 0, carry, body, model, ps, st, st_opt)
end

# TODO: be clear about the scalar nature of the matrices
function gaussloss(means :: Matrix{Float32}, logvars::Matrix{Float32}, sampled::Float32)
    return 0.5 * Lux.mean(logvars .+ (means .- sampled).^2 .* exp.(.-logvars))
end

function randompoint(bounds::Bounds) :: Point
    ((xmin,xmax), (ymin,ymax)) = bounds
    x = rand()*(xmax-xmin) + xmin
    y = rand()*(ymax-ymin) + ymin
    return (x,y)
end

function randomBody(arena :: Arena) :: Body
    (_, _, bounds) = arena
    theta = rand()*tau
    while true
        p = randompoint(bounds)
        if ontrack(p, arena)
            (x,y) = p
            return Body(x, y, theta)
        end
    end
end

function turn(b :: Body, amount)
    # 'amount' can be any real number, but is limited to (-1,1) by a tanh
    turnRate = tau/40;
    return Body(
        b.x,
        b.y,
        b.theta + tanh(amount)*turnRate,
    )
end

function createArena() :: Arena
    # TODO: compute this
    polygon :: Polygon = [
      (-1.2, 2.0),
      (-1.0, 2.2),
      (-0.2, 2.2),
      (-0.2, 3.0),
      ( 0.0, 3.2),
      ( 0.2, 3.0),
      ( 0.2, 2.2),
      ( 1.8, 2.2),
      ( 1.8, 3.0),
      ( 2.0, 3.2),
      ( 2.2, 3.0),
      ( 2.2, 2.2),
      ( 3.8, 2.2),
      ( 3.8, 3.0),
      ( 4.0, 3.2),
      ( 4.2, 3.0),
      ( 4.2, 2.2),
      ( 5.0, 2.2),
      ( 5.2, 2.0),
      ( 5.0, 1.8),
      ( 4.2, 1.8),
      ( 4.2, 0.2),
      ( 5.0, 0.2),
      ( 5.2, 0.0),
      ( 5.0,-0.2),
      ( 4.2,-0.2),
      ( 4.2,-1.0),
      ( 4.0,-1.2),
      ( 3.8,-1.0),
      ( 3.8,-0.2),
      ( 2.2,-0.2),
      ( 2.2,-1.0),
      ( 2.0,-1.2),
      ( 1.8,-1.0),
      ( 1.8,-0.2),
      ( 0.2,-0.2),
      ( 0.2,-1.0),
      ( 0.0,-1.2),
      (-0.2,-1.0),
      (-0.2,-0.2),
      (-1.0,-0.2),
      (-1.2,-0.0),
      (-1.0, 0.2),
      (-0.2, 0.2),
      (-0.2, 1.8),
      (-1.0, 1.8),
      (-1.2, 2.0),
    ]
    inner :: Polygon = [
      (0.2, 1.8),
      (1.8, 1.8),
      (1.8, 1.4),
      (2.0, 1.2),
      (2.2, 1.4),
      (2.2, 1.8),
      (3.8, 1.8),
      (3.8, 0.2),
      (2.2, 0.2),
      (2.2, 0.6),
      (2.0, 0.8),
      (1.8, 0.6),
      (1.8, 0.2),
      (0.2, 0.2),
      (0.2, 1.8),
    ]
    xmin = min(minimum(p -> p[1], polygon), minimum(p -> p[1], inner))
    xmax = max(maximum(p -> p[1], polygon), maximum(p -> p[1], inner))
    ymin = min(minimum(p -> p[2], polygon), minimum(p -> p[2], inner))
    ymax = max(maximum(p -> p[2], polygon), maximum(p -> p[2], inner))
    return (polygon, inner, ((xmin, xmax), (ymin, ymax)))
end

function disable_echo()
    backup_termios = TERMIOS.termios()
    TERMIOS.tcgetattr(stdin, backup_termios)
    termios = TERMIOS.termios()
    TERMIOS.tcgetattr(stdin, termios)
     # Disable echo
    termios.c_lflag &= ~TERMIOS.ECHO
    TERMIOS.tcsetattr(stdin, TERMIOS.TCSANOW, termios)
    return (termios, backup_termios)
end

@timeit to "draw scene" function draw_scene(arena :: Arena, bodies :: Vector{Body}, ages :: Vector{Int64})
    width = 121
    height = 44
    (outer, inner, _) = arena
    canvas = UnicodePlots.BrailleCanvas(height, width, origin_y=-1.2, origin_x=-1.2, height=4.4, width=6.4)
    for (before, after) in zip(outer[1:end-1], outer[2:end])
        UnicodePlots.lines!(canvas, before[1], before[2], after[1], after[2]; color=:cyan)
    end
    for (before, after) in zip(inner[1:end-1], inner[2:end])
        UnicodePlots.lines!(canvas, before[1], before[2], after[1], after[2]; color=:cyan)
    end
    colors=get(ColorSchemes.hawaii,ages,(0.0,500.0))
    for i in eachindex(bodies)
        b = bodies[i]
        c = colors[i]
        col = (round(Int,c.r *255), round(Int,c.g * 255), round(Int,c.b * 255))
        UnicodePlots.points!(canvas, b.x, b.y, color=col)
    end
    plt = UnicodePlots.Plot(canvas)
    # UnicodePlots.scatterplot!(plt, [b.x for b in bodies], [b.y for b in bodies], width=80, height=height, color=colors)
    return (plt, (width, height))
end

function ontrack(p, arena :: Arena)
    (outer, inner, _) = arena
    return PolygonOps.inpolygon(p, outer) == 1 && PolygonOps.inpolygon(p, inner) == 0
end

function sample(means::Matrix{Float32},logvars::Matrix{Float32})::Matrix{Float32}
    sigma = exp.(logvars*0.5)
    return Float32.(randn(size(means))).*sigma + means
end

function moveForward(rng :: Random.AbstractRNG, b::Body)
    speed = 0.05;

    return Body(
        b.x + speed*sin(b.theta), # + Random.randn(rng) * 0.001,
        b.y + speed*cos(b.theta), # + Random.randn(rng) * 0.001,
        b.theta,
    )
end

@timeit to "update car" function updatecar(rng, agent::Car, sensorParams, arena :: Arena)
    sensors = reshape(sensorValues(agent.body, sensorParams, arena),:,1)
    original_carry = agent.carry
    inputs = (sensors, original_carry)
    (means, logvars, carry), st = agent.model(inputs, agent.parameters, agent.state)
    agent.carry = carry
    agent.state = st
    sampled = sample(means, logvars)
    output = sampled[1, 1]
    if isnan(output)
        # we take a zero-tolerance approach to NaNs here - if you output one
        # you are immediately teleported outside the arena and die.
        agent.body = Body(-1000.0,-1000.0,0.0)
        # Avoid propagating the NaNs elsewhere
        output = 0
    end
    agent.body = moveForward(rng, turn(agent.body,output))
    return (sensors, original_carry, means, logvars, output)
end


function replicatecarry(source :: Carry, target :: Carry)
    function replicateone(a, b)
        (s_memory, s_hidden_state) = a
        (t_memory, t_hidden_state) = b
        t_memory .= s_memory
        t_hidden_state .= s_hidden_state
    end
    for (a,b) in zip(source, target)
        replicateone(a, b)
    end
end

@timeit to "replicate car" function replicatecar(rng, source :: Car, target :: Car, arena :: Arena)
    if Random.rand(rng) < 0.01
        input_dims = target.model.lstm_cell.in_dims
        car = Car(rng, Float32.(exp(-2.0-Random.rand(rng, Float32)*5.0)), input_dims, arena)
        replicatecar(rng, car, target, arena)
        return true
    else
        target.model = source.model
        replicatecarry(source.carry, target.carry)
        replicateparams(source.parameters, target.parameters)
        replicateparams(source.state, target.state)
        replicateparams(source.optimiser_state, target.optimiser_state)
        target.lineage = source.lineage
        target.body = source.body
        return false
    end

end


function sensorPoints(b::Body, sensorParams :: Vector{Polar}) :: Vector{Point}
    # for a possible future performance optimisation - only check the endpoints
    # are within the arena, not the whole line.
    function pointFromParams(length,angle)
        x0 = b.x
        y0 = b.y
        x1 = x0 + length*sin(angle + b.theta)
        y1 = y0 + length*cos(angle + b.theta)
        return (x1,y1)
    end
    return [pointFromParams(length,angle) for (length,angle) in sensorParams]
end

function sensorValues(b::Body, sensorParams :: Vector{Polar}, arena :: Arena) :: Vector{Float32}
    points = sensorPoints(b, sensorParams)
    return [!ontrack(p, arena) for p in points]
end

function replicateparams(source :: NamedTuple, target :: NamedTuple)
    function replicatekey(key)
        if !haskey(target, key)
            return deepcopy(getfield(source, key))
        end
        a = getfield(source, key)
        b = getfield(target, key)
        if isa(a,NamedTuple) && isa(b, NamedTuple)
            return replicateparams(a, b)
        elseif isa(a, Array) && isa(b, Array)
            getfield(target, key) .= getfield(source, key)
        else
            return deepcopy(a)
        end
    end
    return NamedTuple{keys(source)}(map(replicatekey, keys(source)))
end


# @timeit to "train_mimic" function train_mimic(agent, inputs :: Vector{Float32}, targets :: Prob)
#     (probs, st1) = network_forward(agent.model.network,inputs, agent.parameters, agent.state, agent.memory.network)
#     loss = divergence_forward(agent.model.mimic_loss, (probs.means, probs.logvars), targets, agent.memory.mimic_loss)
#     divergence_back(agent.model.mimic_loss, Float32(1.0), agent.gradients.mimic_loss, agent.memory.mimic_loss)
#     network_back(agent.model.network, agent.gradients.mimic_loss, agent.parameters, agent.gradients.network, agent.memory.network)
#     return (loss, probs)
# end

# function mimic(agent::Car, params :: AgentParams, arena :: Arena, trajectory::Array{Tuple{Prob,Body}})
#     mid = length(trajectory) ÷ 2
#     if mid == 0
#         return
#     end
#     original_feedback = copy(agent.feedback_nodes)
#     warmup = trajectory[1:mid]
#     training = trajectory[mid+1:end]
#     # replay trajectory to warm up feedback_nodes
#     for (_, body) in warmup
#         sensors = sensorValues(body, params, arena)
#         inputs = [sensors; agent.feedback_nodes*1.0]
#         prob, _ = network_forward(agent.model.network, inputs, agent.parameters, agent.state, agent.memory.network)
#         outputs = sample(prob.means, prob.logvars)
#         feedback = outputs[2:end]
#         agent.feedback_nodes .= update_feedback(agent.feedback_nodes, feedback)
#     end
#     for (prob, body) in training
#         sensors = sensorValues(body, params, arena)
#         inputs :: Vector{Float32} = [sensors; agent.feedback_nodes*1.0]
#         # train on last step of trajectory
#         (_,probs) = train_mimic(agent, inputs, prob)
#         outputs = sample(probs.means, probs.logvars)
#         feedback = outputs[2:end]
#         agent.feedback_nodes .= update_feedback(agent.feedback_nodes, feedback)
#     end
#     # revert feedback_nodes for the agent
#     agent.feedback_nodes .= original_feedback
# end

function cars()
    Base.start_reading(stdin)
    started = time_ns()
    arena = createArena()
    sensorParams :: Vector{Polar} = [
        (d, a*tau)
        for d in [0.1, 0.2, 0.3, 0.4, 0.5]
        for a in [0.25,0.15,0.05,-0.05,-0.15,-0.25]
    ]
    pop_size = 500
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    agents = [Car(rng, Float32.(exp(-2.0-Random.rand(rng, Float32)*5.0)), length(sensorParams), arena) for _ in 1:pop_size]
    history :: Vector{Vector{Tuple{Tuple{Sensors,Sampled},Carry}}} = [[] for _ in 1:pop_size]
    prev = time_ns()
    last_print = 0
    tpf = 0.001
    parents = [i for i in 1:length(agents)]
    frame = 0
    realtime = false
    target_fps = 30
    expectancy = 0.0
    MAX_HISTORY = 400
    while true
        Threads.@threads for k in 1:length(agents)
            agent = agents[k]
            agent.age += 1
            agent.lineage += 1
            (sensors, carry, _, _, sampled) = updatecar(rng, agent, sensorParams, arena)
            push!(history[k], ((sensors,sampled), carry))
            if length(history[k]) > MAX_HISTORY
                popfirst!(history[k])
            end
        end

        alive = [ontrack((agent.body.x, agent.body.y), arena) for agent in agents]

        if !any(alive)
            for i in 1:length(agents)
                agents[i].body = randomBody(arena)
                alive[i] = true
            end
        end

        Threads.@threads for i in 1:length(agents)
            if alive[i]
                parents[i] = i
            else
                k = i
                neighbour = mod1(k+rand([-1,1]), length(agents))
                while !alive[neighbour]
                    k = neighbour
                    neighbour = mod1(k+rand([-1,1]), length(agents))
                end
                @assert alive[neighbour]
                if expectancy == 0.0
                    expectancy = agents[i].age
                else
                    expectancy = 0.999 * expectancy + 0.001 * agents[i].age
                end
                agents[i].age = 0
                new = replicatecar(rng, agents[neighbour], agents[i], arena)
                if !new
                    parents[i] = neighbour
                    history[i] = copy(history[neighbour])
                else
                    parents[i] = 0
                    history[i] = []
                end
            end
        end

        tasks = []
        for k in 1:length(agents)
            if (frame + k) % (MAX_HISTORY ÷ 4) == 0
                t = Threads.@spawn train(agents[k], history[k])
                push!(tasks, t)
            end
        end
        for t in tasks
            Threads.wait(t)
        end

        current = time_ns()
        if current - last_print > 0.05e9
            ages = [agent.age for agent in agents]
            (plt, (_, _)) = draw_scene(arena, [agent.body for agent in agents], ages)
            if to.enabled
                io = PipeBuffer()
                show(IOContext(io), to)
                profiling = string(read(io, String), "\n")
            else
                profiling = ""
            end

            chart = Base.string(plt, color=true)
            mean_age = sum(ages) / length(agents)
            max_age = maximum(ages)
            longest_lineage = maximum([agent.lineage for agent in agents])
            elapsed = current - started
            full_fps =1/(elapsed/(frame*1e9))
            is_realtime = realtime ? "true" : "false"
            summary = @sprintf "\033[K%8.1ffps mean: %7.1ffps age: %6.1f max age: %6d longest lineage: %6d frame: %8d realtime %s life: %6.1f" (1/(tpf/1.0e9)) full_fps mean_age max_age longest_lineage frame is_realtime expectancy
            hist = Base.string(UnicodePlots.histogram(ages, nbins=10, closed=:left, xscale=:log10))
            output = string(chart, "\n", profiling, summary, "\n", hist, "\n")
            lines = countlines(IOBuffer(output))
            print(output)
            print("\033[s") # save cursor
            print("\033[J") # clear to end of screen
            print(string("\033[",lines,"A"))
            last_print = current
        end
        diff = current - prev
        target_step = prev + 1/target_fps * 1e9
        bb = bytesavailable(stdin)
        if bb > 0
            data = read(stdin, bb)
            if data[1] == UInt(32)
                realtime = !realtime
            end
        end
        if realtime && current < target_step
            sleep((target_step - current)/1e9)
        end
        seconds = diff/1.0e9
        alpha = 1 - exp(-0.001*seconds)
        tpf = tpf * alpha + (1 - alpha) * diff
        prev = current
        frame += 1
    end
end

const F_GETFL = Cint(3)
const F_SETFL = Cint(4)
const O_NONBLOCK = Cint(0o00004000)

function main()
    atexit(cleanup)
    s :: RawFD = RawFD(Base.Core.Integer(0))
    flags = ccall(:fcntl, Cint, (RawFD, Cint, Cint...), s, F_GETFL)
    flags2=flags | O_NONBLOCK
    println()
    (_, backup_termios) = disable_echo()
    ccall(:fcntl, Cint, (RawFD, Cint, Cint...), s, F_SETFL, flags2)
    print("\033[?25l") # hide cursor
    Base.exit_on_sigint(false)
    game = ARGS[1]
    if length(ARGS) > 1
        if ARGS[2] == "profile"
            TimerOutputs.enable_timer!(to)
        end
    end
    try
        if game == "cars"
            cars()
        elseif game == "animals"
            animals()
        elseif game == "grads"
            grads()
        else
            cars()
        end
    catch e
        if isa(e, TaskFailedException)
            e = e.task.exception
        end
        if isa(e, Core.InterruptException)
            println("\033[uexiting")
        else
            throw(e)
        end
    finally
        TERMIOS.tcsetattr(stdin, TERMIOS.TCSANOW, backup_termios)
        cleanup()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
