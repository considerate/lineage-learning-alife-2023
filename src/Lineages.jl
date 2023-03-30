module Lineages

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
import Colors
import GLMakie
import Dates

const tau=2*pi
const Polygon = Vector{StaticArrays.SVector{2, Float32}}
const Point = Tuple{Float32, Float32}
const Polar = Tuple{Float32, Float32}
const Bounds = Tuple{Point, Point}
const Arena = Tuple{Polygon, Polygon, Bounds}

const Prob = Tuple{Vector{Float32}, Vector{Float32}}
const Sensors = Matrix{Float32}
const Carry = Tuple{Tuple{Matrix{Float32}, Matrix{Float32}},Tuple{Matrix{Float32}, Matrix{Float32}},Tuple{Matrix{Float32}, Matrix{Float32}}}
const Sampled = Int
const Parent = Int64

to = TimerOutput()
TimerOutputs.disable_timer!(to)


function cleanup()
    print("\033[?25h") # show cursor
end

struct Body
    x::Float32
    y::Float32
    theta::Float32
end

struct CarModel <: Lux.AbstractExplicitContainerLayer{(:lstm_cell, :lstm_cell2, :lstm_cell3, :dense, :motors)}
    lstm_cell:: Lux.LSTMCell
    lstm_cell2:: Lux.LSTMCell
    lstm_cell3:: Lux.LSTMCell
    dense :: Lux.Dense
    motors :: Lux.Dense
end


# A Car decides:
# - what angle to turn using a softmaxed probability
mutable struct Car
    age :: Int64
    lineage :: Int64
    carry :: Carry
    body::Body
    model::CarModel
    parameters::NamedTuple
    state::NamedTuple
    optimiser_state :: NamedTuple
    last_died :: Int64
end

# Run one step through the model
function (model::CarModel)(inputs :: Tuple{AbstractMatrix, AbstractMatrix, Carry}, ps :: NamedTuple, st :: NamedTuple)
    (x, x2, (carry,carry2,carry3)) = inputs
    z, st_dense = model.dense(x, ps.dense, st.dense)
    (y, new_carry), st_lstm = model.lstm_cell((x2, carry), ps.lstm_cell, st.lstm_cell)
    (y2, new_carry_2), st_lstm2 = model.lstm_cell2((y, carry2), ps.lstm_cell2, st.lstm_cell2)
    (y3, new_carry_3), st_lstm3 = model.lstm_cell3((y2, carry3), ps.lstm_cell3, st.lstm_cell3)
    st = merge(st, (lstm_cell=st_lstm, lstm_cell2=st_lstm2, lstm_cell3=st_lstm3))
    mid = z .+ y3
    motors, st_motors = model.motors(mid, ps.motors, st.motors)
    st = merge(st, (motors=st_motors, dense=st_dense))
    return (Lux.logsoftmax(motors), (new_carry, new_carry_2, new_carry_3)), st
end

function sequence_loss(model :: CarModel, initialcarry :: Carry, sequence :: Vector{Tuple{Matrix{Float32}, Matrix{Float32}, Int}}, ps :: NamedTuple, st :: NamedTuple)
    carry = initialcarry
    loss = 0.0
    for (sensors, sensors2, sampled) in sequence
        (motors, carry), st = model((sensors, sensors2, carry), ps, st)
        if !isfinite(loss)
            println(means, logvars, sampled)
            error("Infinite loss")
        end
        loss = loss + logloss(motors, sampled)
    end
    return loss, st
end

function train(agent, history :: Vector{Tuple{Tuple{Matrix{Float32}, Int}, Carry}})
    if length(history) == 0
        return (0.0, 0f0, 0f0)
    end
    (_, carry) = history[1]
    sequence = first.(history)
    wrt = (params=agent.parameters, inputs=[(x,x,sampled) for (x,sampled) in sequence])
    (loss, st), back = Zygote.pullback(x -> sequence_loss(agent.model, carry, x.inputs, x.params, agent.state), wrt)
    agent.state = st
    grads = back((1.0, nothing))[1]
    a = 0.0f0
    b = 0.0f0
    for (g, g2, _) in grads.inputs
        a += sum(g .* g)
        b += sum(g2 .* g2)
    end
    (st_opt, ps) = Optimisers.update!(agent.optimiser_state, agent.parameters, grads.params)
    agent.optimiser_state = st_opt
    agent.parameters = ps
    return (loss, sqrt(a), sqrt(b))
end
function newcarry(rng, sizes, batchsize=1) :: Carry
    (a,b,c) = sizes
    return (
            (Lux.zeros32(rng, a, batchsize), Lux.zeros32(rng, a, batchsize)),
            (Lux.zeros32(rng, b, batchsize), Lux.zeros32(rng, b, batchsize)),
            (Lux.zeros32(rng, c, batchsize), Lux.zeros32(rng, c, batchsize)),
           )
end

function scaled_glorot(rng, dims :: Integer...)
    return Lux.glorot_uniform(rng, dims...; gain=0.5f0)
end

function Car(rng, learning_rate, sensorSize, motorSize, arena, batchsize=1)
    (a,b,c) = (40,30,20)
    model = CarModel(Lux.LSTMCell(sensorSize => a, use_bias=true),
                     Lux.LSTMCell(a => b, use_bias=true),
                     Lux.LSTMCell(b => c, use_bias=true),
                     Lux.Dense(sensorSize => c, Lux.relu, init_weight=scaled_glorot),
                     Lux.Dense(c => motorSize, init_weight=scaled_glorot),
                    )
    (ps, st) = Lux.setup(rng, model)
    body = randomBody(rng, arena)
    st_opt = Optimisers.setup(Optimisers.Descent(learning_rate), ps)
    carry = newcarry(rng, (a,b,c), batchsize)
    return Car(0, 0, carry, body, model, ps, st, st_opt, -10000)
end

function logloss(activations :: Matrix{Float32}, sampled::Int) :: Float32
    return -activations[sampled, 1]
end

function sampleDiscrete(rng, activations :: Matrix{Float32})
    probs = exp.(activations)
    r = Random.rand(rng)
    for i in 1:length(probs)
        r -= probs[i, 1]
        if r<=0.0
            return i
        end
    end
    return length(probs)
end

# TODO: be clear about the scalar nature of the matrices
function gaussloss(means :: Matrix{Float32}, logvars::Matrix{Float32}, sampled::Float32)
    return 0.5 * Lux.mean(logvars .+ (means .- sampled).^2 .* exp.(.-logvars))
end

function randompoint(rng,bounds::Bounds) :: Point
    ((xmin,xmax), (ymin,ymax)) = bounds
    x = Random.rand(rng)*(xmax-xmin) + xmin
    y = Random.rand(rng)*(ymax-ymin) + ymin
    return (x,y)
end

function randomBody(rng, arena :: Arena) :: Body
    (_, _, bounds) = arena
    theta = Random.rand(rng)*tau
    while true
        p = randompoint(rng, bounds)
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

function createLetterArena() :: Tuple{Arena, Int64, Function, Int64}
    outer :: Polygon  = [
        (-0.2, -0.2),
        (-0.2,  2.2),
        ( 0.2,  2.2),
        ( 1.0,  1.7),
        ( 1.8,  2.2),
        ( 2.2,  2.2),
        ( 2.2, -0.2),
        (-0.2, -0.2),
    ]
    inner :: Polygon = [
        (0.2, 0.2),
        (0.2, 1.8),
        (1.0, 1.3),
        (1.8, 1.8),
        (1.8, 0.2),
        (0.2, 0.2),
    ]
    xmin = min(minimum(p -> p[1], outer), minimum(p -> p[1], inner))
    xmax = max(maximum(p -> p[1], outer), maximum(p -> p[1], inner))
    ymin = min(minimum(p -> p[2], outer), minimum(p -> p[2], inner))
    ymax = max(maximum(p -> p[2], outer), maximum(p -> p[2], inner))
    arena = (outer, inner, ((xmin, xmax), (ymin, ymax)))
    return arena, 60, _ -> 1, 1
end

function createArena() :: Tuple{Arena, Int64, Function, Int64}
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
    arena = (polygon, inner, ((xmin, xmax), (ymin, ymax)))
    function get_section(body)
        midy = 1.0
        x1 = 1.0
        x2 = 3.0
        if body.y < midy
            if body.x < x1
                return 1
            elseif body.x < x2
                return 2
            else
                return 3
            end
        else
            if body.x < x1
                return 4
            elseif body.x < x2
                return 5
            else
                return 6
            end
        end

    end
    return arena, 180, get_section, 6
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

function sample(rng, means::Matrix{Float32},logvars::Matrix{Float32})::Tuple{Matrix{Float32}, Matrix{Float32}}
    sigma = exp.(logvars.*Float32(0.5))
    normals = Random.randn(rng, Float32, size(means))
    return (normals, normals.*sigma .+ means)
end

function moveForward(rng :: Random.AbstractRNG, b::Body)
    speed = 0.05;

    return Body(
        b.x + speed*sin(b.theta), # + Random.randn(rng) * 0.001,
        b.y + speed*cos(b.theta), # + Random.randn(rng) * 0.001,
        b.theta,
    )
end

@timeit to "update car" function updatecar(rng, agent::Car, sensorParams, motorParams, arena :: Arena)
    values = sensorValues(agent.body, sensorParams, arena)
    sensors = reshape(values,length(values),1)
    original_carry = agent.carry
    inputs = (sensors, sensors, original_carry)
    (motors, carry), st = agent.model(inputs, agent.parameters, agent.state)
    agent.carry = carry
    agent.state = st
    if !all(map(isfinite, motors))
        println(motors)
        error("Non-finite motor outputs")
    end
    sampled = sampleDiscrete(rng, motors)
    output = motorParams[sampled]
    agent.body = moveForward(rng, turn(agent.body,output))
    return (sensors, original_carry, motors, sampled)
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
    if Random.rand(rng) < 0.0
        input_dims = target.model.lstm_cell.in_dims
        car = Car(rng, Float32.(exp(-2.0-Random.rand(rng, Float32)*5.0)), input_dims, arena)
        replicatecar(rng, car, target, arena)
        return true
    else
        target.model = source.model
        replicatecarry(source.carry, target.carry)
        replicateparams(source.parameters, target.parameters)
        replicateparams(source.state, target.state)
        # replicateparams(source.optimiser_state, target.optimiser_state)
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

function random_lr(rng)
    return Float32(exp(-4.0)) # selected by fair dice roll
    # return Float32.(exp(-3.0-Random.rand(rng, Float32)*5.0))
end

function car_polygon(b)
    return GLMakie.Polygon([
      GLMakie.Point2f(
                      b.x + 0.05*sin(b.theta),
                      b.y + 0.05*cos(b.theta),
                     ),
      GLMakie.Point2f(
                      b.x + 0.015*sin(b.theta + tau/3),
                      b.y + 0.015*cos(b.theta + tau/3),
                     ),
      GLMakie.Point2f(
                      b.x + 0.015*sin(b.theta - tau/3),
                      b.y + 0.015*cos(b.theta - tau/3),
               ),
      GLMakie.Point2f(
                      b.x + 0.05*sin(b.theta),
                      b.y + 0.05*cos(b.theta),
               ),
     ])
end

function traillength(trail)
    return sum([length(t) for t in trail])
end

function to_poly(ps)
    GLMakie.Polygon([GLMakie.Point2f(p[1], p[2]) for p in ps])
end

function plot_cars(arena, trails, agents, sensorParams)
    (polygon, inner, bounds) = arena
    (xlims, ylims) = bounds
    (xmin, xmax) = xlims
    (ymin, ymax) = ylims
    width = xmax - xmin + 0.2
    height = ymax - ymin + 0.2
    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis(fig[1, 1], aspect=width/height)
    GLMakie.hidedecorations!(ax)  # hides ticks, grid and lables
    GLMakie.hidespines!(ax)
    GLMakie.limits!(ax, (xmin - 0.1, xmax + 0.1), (ymin - 0.1, ymax + 0.1))
    makie_bodies = GLMakie.Observable([agent.body for agent in agents])
    makie_cars = GLMakie.lift(xs -> car_polygon.(xs), makie_bodies)
    makie_sensors = GLMakie.Observable(sensorPoints(agents[1].body, sensorParams))
    makie_trails = GLMakie.Observable(trails)
    black_segments = GLMakie.Observable(GLMakie.Point2f[])
    red_segments = GLMakie.Observable(GLMakie.Point2f[])
    function update_trails(ts)
        empty!(black_segments[])
        empty!(red_segments[])
        for trail in ts
            for (k,segments) in enumerate(trail)
                for (before, after) in segments
                    if k == length(trail)
                        push!(black_segments[], GLMakie.Point2f(before.x, before.y))
                        push!(black_segments[], GLMakie.Point2f(after.x, after.y))
                    else
                        push!(red_segments[], GLMakie.Point2f(before.x, before.y))
                        push!(red_segments[], GLMakie.Point2f(after.x, after.y))
                    end
                end
            end
        end
        GLMakie.notify(red_segments)
        GLMakie.notify(black_segments)
    end
    GLMakie.on(update_trails, makie_trails)
    update_trails(makie_trails[])
    GLMakie.poly!(ax,to_poly(polygon);color=:transparent,strokewidth=5,strokecolor=:gray)
    GLMakie.poly!(ax,to_poly(inner);color=:transparent,strokewidth=5,strokecolor=:gray)
    GLMakie.linesegments!(ax, red_segments, strokewidth = 1, color = Colors.RGBA(1.0,0,0,0.1))
    GLMakie.linesegments!(ax, black_segments, strokewidth = 2, color = Colors.RGBA(0,0,0,0.2))
    GLMakie.poly!(ax, makie_cars,color=GLMakie.Cycled(1))
    GLMakie.scatter!(ax, makie_sensors,color=GLMakie.Cycled(2))
    return (fig, makie_bodies, makie_sensors, makie_trails)
end

function cars()
    GLMakie.activate!()
    Base.start_reading(stdin)
    started = time_ns()
    # (arena, MAX_TRAIL, section, sections) = createLetterArena()
    (arena, MAX_TRAIL, section, sections) = createArena()
    sensorParams :: Vector{Polar} = [
        (d, a*tau)
        for d in [0.1, 0.2, 0.3, 0.4, 0.5]
        for a in [0.25,0.15,0.05,-0.05,-0.15,-0.25]
    ]
    motorParams = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]
    pop_size = 500
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    agents = [Car(rng, random_lr(rng), length(sensorParams), length(motorParams), arena) for _ in 1:pop_size]
    trails :: Vector{Vector{Vector{Tuple{Body,Body}}}} = [[[]] for _ in 1:pop_size]
    (fig, makie_bodies, makie_sensors, makie_trails) = plot_cars(arena, trails, agents, sensorParams)
    history :: Vector{Vector{Tuple{Tuple{Sensors,Sampled},Carry, Body, Body}}} = [[] for _ in 1:pop_size]
    last_steps :: Vector{Tuple{Body, Body}} = [(Body(0,0,0), Body(0,0,0)) for _ in 1:pop_size]
    prev = time_ns()
    last_print = 0
    tpf = 0.001
    parents = [i for i in 1:length(agents)]
    realtime = false
    target_fps = 30
    expectancy = 0.0
    MAX_HISTORY = 100
    empty :: Matrix{Float32} = Float32.([0.0 0.0])
    showwindow = true
    frames = 1:100000
    deathsperframe = 0.0
    halflife = 250.0 # frames
    decay = 1 - 2^(-1/halflife)
    if !isdir("runs")
        mkdir("runs")
    end
    rundir = string("runs/", Dates.now())
    mkdir(rundir)
    mkdir(string(rundir, "/frames"))
    exitcolumns = join([string("exit_",c) for c in 1:sections], ",")
    crashcolumns = join([string("crash_",c) for c in 1:sections], ",")
    columns = string("frame,deaths,deaths_per_frame,max_age,mean_age,loss", ",", exitcolumns, ",", crashcolumns)
    if showwindow
        display(GLMakie.Screen(), fig)
        loop = f -> open(string(rundir,"/mimicry.csv"), "w+") do io
            println(io,columns)
            foreach(x -> f(x, io), frames)
        end
    else
        loop = f -> open(string(rundir,"/mimicry.csv"), "w+") do io
            println(io, columns)
            GLMakie.record(x -> f(x,io), fig, string(rundir,"/animation.mp4"), frames; framerate=12, compression=20)
        end
    end
    if !showwindow
        filename = string(rundir, "/frames/","frame-",0,".png")
        GLMakie.save(filename, fig)
    end
    loop() do frame, csv
        motorss :: Vector{Matrix{Float32}} = [empty for _ in agents]
        Threads.@threads for k in 1:length(agents)
            agent = agents[k]
            agent.age += 1
            agent.lineage += 1
            body = agent.body
            (sensors, carry, motors, sampled) = updatecar(rng, agent, sensorParams, motorParams, arena)
            motorss[k] = exp.(motors)
            last_steps[k] = (body, agent.body)
            push!(history[k], ((sensors,sampled), carry, body, agent.body))
            push!(trails[k][end], (body, agent.body))
            if length(history[k]) > MAX_HISTORY
                popfirst!(history[k])
            end
            if traillength(trails[k]) > MAX_TRAIL
                while length(trails[k][1]) == 0
                    popfirst!(trails[k])
                end
                if length(trails[k]) > 0
                    popfirst!(trails[k][1])
                end
            end
        end
        alive = [ontrack((agent.body.x, agent.body.y), arena) for agent in agents]
        deaths = sum([ a ? 0 : 1 for a in alive])
        deathsperframe = deaths * decay  + deathsperframe * (1 - decay)
        exits = [0 for _ in 1:sections]
        crashes = [0 for _ in 1:sections]
        cells = [0 for _ in 1:sections]
        for (agent, (before,after),survived) in zip(agents,last_steps,alive)
            cells[section(after)] += 1
            if agent.last_died < frame - 30
                if !survived
                    crashes[section(before)] += 1
                else
                    if section(before) != section(after)
                        exits[section(before)] += 1
                    end
                end
            end
        end

        if !any(alive)
            for i in 1:length(agents)
                agents[i].body = randomBody(rng, arena)
                agents[i].age = 0
                agents[i].last_died = frame
                history[i] = []
                trails[i] = [[]]
                alive[i] = true
            end
        end

        Threads.@threads for i in 1:length(agents)
            if alive[i]
                parents[i] = i
            else
                k = i
                neighbour = Random.rand(rng, 1:length(agents)) # mod1(k+rand([-1,1]), length(agents))
                while !alive[neighbour]
                    k = neighbour
                    # neighbour = mod1(k+rand([-1,1]), length(agents))
                    neighbour = Random.rand(rng, 1:length(agents)) # mod1(k+rand([-1,1]), length(agents))
                end
                @assert alive[neighbour]
                if expectancy == 0.0
                    expectancy = agents[i].age
                else
                    expectancy = 0.999 * expectancy + 0.001 * agents[i].age
                end
                replicatecar(rng, agents[neighbour], agents[i], arena)
                agents[i].age = 0
                agents[i].last_died = frame
                history[i] = copy(history[neighbour])
                if length(trails[i][end]) > 0
                    push!(trails[i], [])
                end
            end
        end


        tasks = []
        losses = [0.0 for _ in agents]
        for k in 1:length(agents)
            if (frame + k) % (MAX_HISTORY ÷ 10) == 0
                t = Threads.@spawn begin
                    (l, a, b) = train(agents[k], [(values, carry) for (values, carry, _, _) in history[k]])
                    losses[k] += l
                    return (a,b)
                end
                push!(tasks, t)
            end
        end
        results = [fetch(t) for t in tasks]
        dense_grads = Lux.mean([a for (a,_) in results])
        lstm_grads = Lux.mean([b for (_,b) in results])

        # Update our plot state
        makie_bodies[] = [agent.body for agent in agents]
        makie_sensors[] = sensorPoints(agents[1].body, sensorParams)
        makie_trails[] = copy(trails)
        yield()
        if frame % 100 == 0 && !showwindow
            filename = string(rundir, "/frames/","frame-",frame,".png")
            GLMakie.save(filename, fig)
        end

        current = time_ns()
        ages = [agent.age for agent in agents]
        longest_lineage = maximum([agent.lineage for agent in agents])
        mean_age = sum(ages) / length(agents)
        max_age = maximum(ages)
        loss = Lux.mean(losses)
        line = join([
                      join([frame, deaths, deathsperframe, max_age, mean_age, loss], ","),
                      join(exits, ","),
                      join(crashes, ","),
                     ],
                    ","
        )
        println(csv, line)
        flush(csv)
        if current - last_print > 0.05e9
            (plt, (_, _)) = draw_scene(arena, [agent.body for agent in agents], ages)
            if to.enabled
                io = PipeBuffer()
                show(IOContext(io), to)
                profiling = string(read(io, String), "\n")
            else
                profiling = ""
            end

            chart = Base.string(plt, color=true)
            elapsed = current - started
            full_fps = 1/(elapsed/(frame*1e9))
            is_realtime = realtime ? "true" : "false"
            summary = @sprintf "\033[K%8.1ffps mean: %7.1ffps age: %6.1f max age: %6d longest lineage: %6d frame: %8d realtime %s life: %6.1f\tloss: %2.3g" (1/(tpf/1.0e9)) full_fps mean_age max_age longest_lineage frame is_realtime expectancy loss
            hist = Base.string(UnicodePlots.histogram(ages, nbins=10, closed=:left, xscale=:log10))
            output = string(chart, "\n", profiling, summary, "\n", cells, "\n", hist, "\n", "\n\n", dense_grads, " : ", lstm_grads, "\n\n")
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
    end
end

const F_GETFL = Cint(3)
const F_SETFL = Cint(4)
const O_NONBLOCK = Cint(0o00004000)

function run()
    atexit(cleanup)
    s :: RawFD = RawFD(Base.Core.Integer(0))
    flags = ccall(:fcntl, Cint, (RawFD, Cint, Cint...), s, F_GETFL)
    flags2=flags | O_NONBLOCK
    println()
    (_, backup_termios) = disable_echo()
    ccall(:fcntl, Cint, (RawFD, Cint, Cint...), s, F_SETFL, flags2)
    print("\033[?25l") # hide cursor
    Base.exit_on_sigint(false)
    try
        cars()
    catch e
        while isa(e, TaskFailedException)
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

function main()
    run()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
