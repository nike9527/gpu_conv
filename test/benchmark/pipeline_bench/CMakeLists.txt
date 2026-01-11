add_test(
  NAME PipelineBenchmark
  COMMAND pipeline_bench
)

set_tests_properties(PipelineBenchmark PROPERTIES
  PASS_REGULAR_EXPRESSION "Triple"
)
