using Test

const TEST_DIR = @__DIR__

function include_test(file::String)
    println("\n=== Running $(file) ===")
    include(joinpath(TEST_DIR, file))
end

# Test lane flags:
# - default: always runs fast sanity checks
# - medium: adds relation extraction coverage
# - full: adds all heavier suites (includes medium)
full_suite = get(ENV, "SWAMMA_TEST_FULL", "0") == "1"
medium_suite = full_suite || get(ENV, "SWAMMA_TEST_MEDIUM", "0") == "1"

# Fast/default suite for local and CI sanity.
include_test("test_attention.jl")
include_test("test_router.jl")
include_test("test_wavepde.jl")
include_test("test_llada_training.jl")
include_test("test_training_padding.jl")
include_test("test_train_chess_reasoning.jl")
include_test("test_reasoning_trainability.jl")

# Medium suite is opt-in and intended for broader but still practical coverage.
if medium_suite
    include_test("test_relation_extraction.jl")
end

# Full suite is opt-in because some tests are heavier / task-specific.
if full_suite
    include_test("test_moet.jl")
    include_test("test_tidar.jl")
end
