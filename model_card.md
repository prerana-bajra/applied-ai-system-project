# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**BeatBuddy 1.0**

---

## 2. Goal / Task

This recommender suggests songs from a small catalog. It tries to match a user's genre, mood, and audio preferences. It is a classroom demo, not a real streaming app.

---

## 3. Data Used

The catalog has 50 songs in `data/songs.csv`. Each song has genre, mood, energy, tempo, valence, danceability, and acousticness. The dataset covers many styles like pop, lofi, rock, EDM, jazz, and classical. Some tastes are missing or only weakly represented, like sad songs or very unusual genres.

---

## 4. Data

The catalog has 50 songs in `data/songs.csv`. It includes genres like pop, lofi, rock, EDM, jazz, hip hop, and classical. It also includes moods like happy, chill, intense, calm, and moody. I changed a few rows to make some songs more similar for testing. Some tastes are still missing, like sad songs and unusual genre combinations.

---

## 5. Strengths

The system works well for users with clear tastes, like Happy Pop, Chill Lofi, or Deep Intense Rock. It does a good job when the song's genre and audio features line up with the user's request. It also handles strong energy or calmness patterns in a way that feels reasonable. In my tests, the top songs often matched the profile I expected.

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

One weakness I found is that the recommender can over-favor songs that match the user's genre and energy, even when the mood is a poor fit. In my experiments, Gym Hero kept showing up for users who wanted Happy Pop because it is still pop and has very similar energy, danceability, and tempo values. The system also does not penalize mood mismatches, so a sad or conflicting mood does not push a song down very much. This can create a filter bubble where upbeat pop songs appear too often and other styles do not get enough chance to surface. The dataset itself also has a limited set of moods, so some user tastes are not represented well.

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

---

## 6. Evaluation Process

I tested these profiles: High-Energy Pop, Chill Lofi, Deep Intense Rock, Conflict: High Energy + Sad, Conflict: Acoustic Raver, and Edge: Unknown Labels + Out of Range. I checked whether the top songs matched what each profile asked for. I also compared the ranking changes after I shifted the scoring weights. The biggest surprise was that Gym Hero stayed high even when the mood was not a good fit.

---

## 7. Intended Use and Non-Intended Use

This system is meant for learning and experimentation. It is good for showing how simple features can drive recommendations. It should not be used as a real music app for important decisions. It should also not be treated as fair or complete taste analysis.

---

## 8. Ideas for Improvement

- Add a penalty when mood does not match.
- Add more genres, moods, and edge-case songs.
- Improve diversity so the same song type does not appear too often.

---

## 9. Personal Reflection

My biggest learning moment was seeing how one small scoring choice can change the whole ranking. I learned that a recommender does not need to be complex to seem useful, but it can still repeat the same patterns over and over.

AI tools helped me organize ideas, test edge cases, and explain the results in simpler words. I still had to double-check the numbers and rankings myself, because the model can sound confident even when a result is not the best match.

What surprised me most was that simple rules can still feel like real recommendations when the songs line up well. Even a basic score can seem smart if the genre and energy are close to what the user asked for.

If I extended this project, I would add mood penalties, more song variety, and a way to rank for diversity instead of just closeness. I would also try a richer explanation system so users can see why a song won.
