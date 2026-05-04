# Hello, world

This is the first post on the new blog. From here on out, I'll be using this space to write about AI, distributed systems, and whatever else I'm building.

## How this blog works

I wanted something dead simple — no Jekyll, no Next.js, no build step. Just markdown.

Here's the whole setup:

- Posts are markdown files in [`/blog/`](https://github.com/roywei/roywei.github.io/tree/main/blog) on GitHub.
- A tiny `js/posts.js` lists them by slug, title, and date.
- `post.html` fetches the markdown and renders it with [marked](https://marked.js.org/) on the client.
- The whole site is still served as static files by GitHub Pages.

To publish a new post I do exactly two things:

1. `git add blog/my-new-post.md`
2. Add one entry to `js/posts.js`

That's it. No CMS. No deploy pipeline. The total weight of the homepage is one CSS file and a single inline analytics snippet.

## What's next

A few things I'd like to write about soon:

- What it's actually like to manage an ML systems team at AWS
- Notes from building [CyberBunny](https://github.com/roywei/cyber-bunny) — a robot car driven by GPT-4o
- Why *Person of Interest* is still the best fictional AI

More soon.
