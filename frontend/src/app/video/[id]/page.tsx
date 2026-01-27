type VideoPageProps = {
  params: Promise<{ id: string }>;
};

type VideoStatus = "processing" | "ready" | "error";

// Mock function - will be replaced with API call
function getVideoStatus(): VideoStatus {
  return "processing";
}

export default async function VideoPage({ params }: VideoPageProps) {
  const { id } = await params;
  const status = getVideoStatus();

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <h1 className="text-2xl font-bold mb-2">Video: {id}</h1>

      {status === "processing" && (
        <div className="text-center">
          <p className="text-zinc-600 dark:text-zinc-400 mb-4">
            Processing your video...
          </p>
          <div className="w-64 h-2 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
            <div className="w-1/2 h-full bg-zinc-900 dark:bg-zinc-100 animate-pulse" />
          </div>
          <p className="mt-2 text-sm text-zinc-500">
            (Status polling will be added in next iteration)
          </p>
        </div>
      )}

      {status === "ready" && (
        <div className="w-full max-w-xl">
          <input
            type="text"
            placeholder="Search for a moment..."
            className="w-full px-4 py-3 border border-zinc-300 dark:border-zinc-700 rounded-lg bg-white dark:bg-zinc-900"
            disabled
          />
          <button
            className="mt-4 w-full px-4 py-3 bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 rounded-lg font-medium disabled:opacity-50"
            disabled
          >
            Search
          </button>

          {/* Results will appear here */}
          <div className="mt-8 grid grid-cols-3 gap-4">
            <div className="aspect-video bg-zinc-200 dark:bg-zinc-800 rounded flex items-center justify-center text-zinc-500">
              Result 1
            </div>
            <div className="aspect-video bg-zinc-200 dark:bg-zinc-800 rounded flex items-center justify-center text-zinc-500">
              Result 2
            </div>
            <div className="aspect-video bg-zinc-200 dark:bg-zinc-800 rounded flex items-center justify-center text-zinc-500">
              Result 3
            </div>
          </div>
        </div>
      )}

      <a
        href="/"
        className="mt-8 text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
      >
        &larr; Back to home
      </a>
    </main>
  );
}
