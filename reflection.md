# Reflection

I tested six profiles: High-Energy Pop, Chill Lofi, Deep Intense Rock, Conflict: High Energy + Sad, Conflict: Acoustic Raver, and Edge: Unknown Labels + Out of Range.

## Pairwise Comments

High-Energy Pop vs Chill Lofi: High-Energy Pop pushed songs like Sunrise City and Rooftop Lights to the top because they match pop, happy, and high energy, while Chill Lofi moved toward Library Rain, Focus Flow, and Midnight Coding because those songs are slower and calmer.

High-Energy Pop vs Deep Intense Rock: Both profiles liked energetic songs, but Deep Intense Rock shifted the top results toward rock tracks like Storm Runner and Thunder Garden because the genre and mood were a better fit.

High-Energy Pop vs Conflict: High Energy + Sad: These two profiles shared the same high-energy shape, so Gym Hero still ranked highly in both cases. The difference was that the sad mood did not lower the score much, which shows the model does not punish mood mismatches.

High-Energy Pop vs Conflict: Acoustic Raver: The Acoustic Raver profile pulled in more electronic songs like Neon Horizon and Electric Bloom, but some calm songs still appeared because the system also rewards acoustic similarity and shared numeric features.

High-Energy Pop vs Edge: Unknown Labels + Out of Range: The edge-case profile did not match any genre or mood cleanly, so the model fell back on raw audio similarity and returned songs with similar numbers instead of songs that clearly matched the request.

Chill Lofi vs Deep Intense Rock: These two profiles were almost opposites. Chill Lofi favored slow, soft songs, while Deep Intense Rock favored louder, faster, heavier songs, so the top results changed almost completely.

Chill Lofi vs Conflict: High Energy + Sad: Chill Lofi ranked calm songs because of the lofi and chill labels, but the conflicting profile ranked energetic pop songs because the energy score mattered more than the sad mood.

Chill Lofi vs Conflict: Acoustic Raver: Both profiles liked songs with acoustic or softer features, but Acoustic Raver also asked for very high energy, so it pulled in more danceable electronic tracks.

Chill Lofi vs Edge: Unknown Labels + Out of Range: The lofi profile had clear labels the model understands, while the edge-case profile used strange labels and extreme numbers, so the edge-case output was less intuitive and more dependent on feature distance.

Deep Intense Rock vs Conflict: High Energy + Sad: Both profiles liked intense music, but Deep Intense Rock rewarded rock songs with matching mood, while the sad profile allowed pop songs to win as long as the energy stayed close.

Deep Intense Rock vs Conflict: Acoustic Raver: Deep Intense Rock favored loud and heavy songs, while Acoustic Raver mixed high energy with a preference for acousticness, so the results became more mixed and less rock-heavy.

Deep Intense Rock vs Edge: Unknown Labels + Out of Range: The rock profile had a clear target, but the edge profile did not, so the model had a much harder time producing a result that felt obviously correct.

Conflict: High Energy + Sad vs Conflict: Acoustic Raver: These two adversarial profiles exposed different weaknesses. The sad profile showed that mood mismatch is barely punished, while the acoustic raver profile showed that the model can be pulled in two directions at once.

Conflict: High Energy + Sad vs Edge: Unknown Labels + Out of Range: The sad profile still produced songs that looked reasonable because the labels were familiar, but the edge profile broke the assumptions more strongly by using unknown genre and mood values.

Conflict: Acoustic Raver vs Edge: Unknown Labels + Out of Range: The acoustic raver profile still gave a somewhat readable result because the model understood the genre and mood words, while the edge profile mostly showed that the system needs more realistic input values to behave well.

One thing that surprised me was how often Gym Hero stayed near the top for Happy Pop users. In plain language, the song keeps showing up because the model says, "This is still pop, and the energy and sound are close enough," even though the mood is not a perfect match.
