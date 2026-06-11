# Launch Kit — comment partager Q-Drop (honnêtement et efficacement)

Tout ce qu'il faut pour promouvoir le projet toi-même, sur les bons canaux,
avec des textes prêts à copier-coller. Chaque texte est 100% factuel —
c'est ce qui convertit le mieux auprès des développeurs, et c'est ce qui
évite de se faire démolir en commentaires.

⚠️ Règle d'or : poste chaque message UNE fois, sur le bon canal, et réponds
aux commentaires. Le spam cross-posté est contre-productif et bannissable.

---

## 1. Hacker News — "Show HN" (le canal #1 pour ce produit)

Poster sur https://news.ycombinator.com/submit
Meilleur créneau : 14h-16h heure de Paris (matin US), mardi-jeudi.

**Titre :**
> Show HN: I implemented the algorithms behind RocksDB, Pinecone and Kafka in readable Python

**URL :** https://github.com/AmineCHABANE/Q-Drop-Marketplace

**Premier commentaire (à poster immédiatement sous ta soumission) :**
> I built 5 reference implementations of core infrastructure algorithms —
> HNSW vector search, LSM-trees, columnar engines, stream windowing, and
> arena allocation — in pure Python with zero dependencies, heavily
> commented, with a 42-test suite.
>
> They're optimized for readability, not speed: the goal is understanding
> what RocksDB/Pinecone/Flink actually do internally, not replacing them.
>
> The code is fully public (fair-code model): free to read and study,
> €9.99 one-time license if you use it commercially. Happy to answer
> questions about any of the implementations.

---

## 2. Reddit (un subreddit à la fois, adapte le ton à chacun)

### r/Python — angle "j'ai construit ça"
**Titre :** I implemented HNSW, LSM-trees, and Kafka-style stream windowing in pure Python (zero deps, fully commented)
**Corps :** explique ce que tu as appris en les écrivant + lien repo. Ne mets PAS le prix en avant — les commentaires le découvriront, c'est OK.

### r/ExperiencedDevs ou r/cscareerquestions — angle interview prep
**Titre :** Studying for system design interviews? I wrote readable reference implementations of the 5 algorithms that come up most
**Corps :** LSM-trees, vector search, windowing, columnar, allocators — avec lien.

### r/programming — seulement si le post HN a bien marché (lien vers le repo)

---

## 3. X / Twitter — thread

**Tweet 1 :**
> Ever wondered what's actually inside RocksDB, Pinecone, or Kafka Streams?
>
> I implemented their core algorithms in readable, commented Python:
> • HNSW vector search
> • LSM-tree storage
> • Columnar analytics
> • Stream windowing + watermarks
> • Arena allocation
>
> 100% of the code is public 🧵

**Tweet 2 :**
> Each module is a few hundred lines, zero dependencies, with references
> to the original papers. Run the 42-test suite yourself:
>
> git clone https://github.com/AmineCHABANE/Q-Drop-Marketplace
> python3 -m unittest discover -s src/tests

**Tweet 3 :**
> Free to read and study. €9.99 once if you use it commercially
> (fair-code, like Sidekiq). All future modules included.
>
> https://aminechabane.github.io/Q-Drop-Marketplace/

---

## 4. LinkedIn — angle apprentissage

> J'ai passé du temps à implémenter les algorithmes au cœur de RocksDB,
> Pinecone et Kafka Streams — en Python lisible et commenté.
>
> Pourquoi ? Parce que lire une implémentation de 300 lignes apprend plus
> que survoler une codebase de 300 000 lignes.
>
> Le code est entièrement public, avec une suite de tests complète.
> Lien en commentaire. [puis poste le lien GitHub en commentaire]

---

## 5. Autres canaux pertinents

- **Lobsters** (lobste.rs) — tag "programming", même angle que HN (il faut une invitation)
- **dev.to** — réécris le README en article "How LSM-trees work, with code"
- **Newsletter Python Weekly / Pycoders** — soumets le repo via leur formulaire
- **Discord/Slack de communautés data engineering** — partage seulement si tu participes déjà

---

## 6. Ce qui fera VRAIMENT décoller les ventes (au-delà du partage)

1. **Articles de blog par module** : "How RocksDB works, explained with 300
   lines of Python" — chaque article amène du SEO durable vers le site.
2. **Nouveaux modules réguliers** : bloom filters, consistent hashing, Raft —
   chaque ajout = une nouvelle occasion de poster, et augmente la valeur de
   la licence "tout inclus à vie".
3. **Répondre vite aux issues GitHub** : un repo vivant convertit mieux.

---

## À NE PAS FAIRE

- ❌ Faux témoignages, faux compteurs, faux prix barrés — détruisent la
  confiance et exposent juridiquement (pratiques commerciales trompeuses, UE).
- ❌ Prétendre que le code remplace RocksDB/FAISS en production.
- ❌ Cross-poster le même texte partout le même jour (filtres anti-spam).
- ❌ Acheter des upvotes/followers.

Le pitch honnête est aussi le pitch le plus fort ici : "tout le code est
public, vérifie avant de payer" est un argument de vente, pas une faiblesse.
