export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold mb-4">Video Moment Finder</h1>
      <p className="text-zinc-600 dark:text-zinc-400 mb-8">
        Paste a YouTube URL to find specific moments
      </p>

      {/* URL Input - to be wired up in next commit */}
      <div className="w-full max-w-xl">
        <input
          type="text"
          placeholder="https://www.youtube.com/watch?v=..."
          className="w-full px-4 py-3 border border-zinc-300 dark:border-zinc-700 rounded-lg bg-white dark:bg-zinc-900"
          disabled
        />
        <button
          className="mt-4 w-full px-4 py-3 bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 rounded-lg font-medium disabled:opacity-50"
          disabled
        >
          Process Video
        </button>
        <p className="mt-2 text-sm text-zinc-500 text-center">
          (Form will be enabled in next iteration)
        </p>
      </div>
    </main>
  );
}
