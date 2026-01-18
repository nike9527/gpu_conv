TEST(TriplePipeline, BasicFlow)
{
    TriplePipeline<DummySlot> pipeline;

    auto *slot = pipeline.acquire_free();
    ASSERT_NE(slot, nullptr);

    pipeline.submit(*slot);
    slot->done = true;

    auto *ready = pipeline.try_collect();
    ASSERT_EQ(ready, slot);

    pipeline.release(*ready);
}
