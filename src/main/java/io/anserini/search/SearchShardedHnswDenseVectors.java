/*
 * Anserini: A Lucene toolkit for reproducible information retrieval research
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.anserini.search;

import com.fasterxml.jackson.core.JsonProcessingException;

import io.anserini.index.ShardInfo;
import io.anserini.search.topicreader.TopicReader;
import io.anserini.search.topicreader.Topics;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;
import org.kohsuke.args4j.spi.StringArrayOptionHandler;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.stream.IntStream;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Collections;



/**
 * Main entry point for sharded HNSW dense vector search.
 */
public final class SearchShardedHnswDenseVectors<K extends Comparable<K>> implements Runnable, Closeable {
  private static final Logger LOG = LogManager.getLogger(SearchShardedHnswDenseVectors.class);

  public static class Args extends HnswDenseSearcher.Args {
    @Option(name = "-topics", metaVar = "[file]", handler = StringArrayOptionHandler.class, required = true, usage = "topics file")
    public String[] topics;

    @Option(name = "-output", metaVar = "[file]", required = true, usage = "output file")
    public String output;

    @Option(name = "-topicReader", usage = "TopicReader to use.")
    public String topicReader = "JsonIntVector";

    @Option(name = "-topicField", usage = "Topic field that should be used as the query.")
    public String topicField = "vector";

    @Option(name = "-hits", metaVar = "[number]", usage = "max number of hits to return")
    public int hits = 1000;

    @Option(name = "-runtag", metaVar = "[tag]", usage = "runtag")
    public String runtag = "Anserini";

    @Option(name = "-format", metaVar = "[output format]", usage = "Output format, default \"trec\", alternative \"msmarco\".")
    public String format = "trec";

    @Option(name = "-options", usage = "Print information about options.")
    public Boolean options = false;
  }

  private final Args args;
  private final List<HnswDenseSearcher<K>> searchers;
  private final List<K> qids = new ArrayList<>();
  private final List<String> queries = new ArrayList<>();
  private final ShardInfo[] shards;
  private final int threadsPerShard;

  /*
   * Constructor for sharded HNSW dense vector search.
   * The caller must provide an identifier for the sharded index defined in ShardInfo.
   * This identifier is used to retrieve the shards from ShardInfo.
   */
  public SearchShardedHnswDenseVectors(Args args) throws IOException {
    this.args = args;
    this.searchers = new ArrayList<>();
    this.shards = ShardInfo.getShardedIndex(args.index);
    this.threadsPerShard = Math.max(args.threads / shards.length, 1);

    LOG.info("============ Initializing {} ============", this.getClass().getSimpleName());
    LOG.info("Found {} shards for identifier: {}", shards.length, args.index);
    LOG.info("Topics: {}", Arrays.toString(args.topics));
    LOG.info("Query generator: {}", args.queryGenerator);
    LOG.info("Encoder: {}", args.encoder);
    LOG.info("Threads: {}", args.threads);
    LOG.info("Threads per shard: {}", threadsPerShard);

    String indexPath = "/store/scratch/v4zhong/.cache/pyserini/indexes/";
    
    for (ShardInfo shard : shards) {
      HnswDenseSearcher.Args searcherArgs = new HnswDenseSearcher.Args();
      searcherArgs.index = indexPath + shard.indexName;
      searcherArgs.encoder = args.encoder;
      searcherArgs.queryGenerator = args.queryGenerator;
      searcherArgs.efSearch = args.efSearch;
      searcherArgs.threads = threadsPerShard;
      searchers.add(new HnswDenseSearcher<>(searcherArgs));
    }

    // We might not be able to successfully read topics for a variety of reasons. Gather all possible
    // exceptions together as an unchecked exception to make initialization and error reporting clearer.
    SortedMap<K, Map<String, String>> topics = new TreeMap<>();
    for (String topicsFile : args.topics) {
      Path topicsFilePath = Paths.get(topicsFile);
      if (!Files.exists(topicsFilePath) || !Files.isRegularFile(topicsFilePath) || !Files.isReadable(topicsFilePath)) {
        Topics ref = Topics.getByName(topicsFile);
        if (ref == null) {
          throw new IllegalArgumentException(String.format("\"%s\" does not refer to valid topics.", topicsFilePath));
        } else {
          topics.putAll(TopicReader.getTopics(ref));
        }
      } else {
        try {
          @SuppressWarnings("unchecked")
          TopicReader<K> tr = (TopicReader<K>) Class
            .forName(String.format("io.anserini.search.topicreader.%sTopicReader", args.topicReader))
            .getConstructor(Path.class).newInstance(topicsFilePath);

          topics.putAll(tr.read());
        } catch (Exception e) {
          throw new IllegalArgumentException(String.format("Unable to load topic reader \"%s\".", args.topicReader));
        }
      }
    }

    // Now iterate through all the topics to pick out the right field with proper
    // exception handling.
    try {
      topics.forEach((qid, topic) -> {
        String query;
        if (args.encoder != null) {
          query = topic.get("title");
        } else {
          query = topic.get(args.topicField);
        }
        assert query != null;
        qids.add(qid);
        queries.add(query);
      });
    } catch (AssertionError | Exception e) {
      throw new IllegalArgumentException(String.format("Unable to read topic field \"%s\".", args.topicField));
    }
  }

  /*
   * Resource management differs from non-sharded:
   * - Original closes single searcher in close()
   * - Sharded version currently closes in run() (should be moved to close())
   * - Needs to handle multiple searcher instances
   * - Must ensure all resources released even on failure
   */
  @Override
  public void close() throws IOException {
    LOG.info("Closing searchers...");
    for (HnswDenseSearcher<K> searcher : searchers) {
      try {
        searcher.close();
        LOG.info("Closed searcher: {}", searcher.getClass().getSimpleName());
      } catch (IOException e) {
        LOG.warn("Error closing searcher: {}", searcher.getClass().getSimpleName(), e);
      }
    }
    LOG.info("All searchers closed.");
  }

  /*
   * run() method implements parallel shard processing:
   * - Uses IntStream.parallel() to process shards concurrently
   * - Each shard gets dedicated thread allocation to prevent oversubscription
   * - Writes results immediately per shard instead of merging
   * - Closes searchers after use to manage memory
   * Key difference from non-sharded: Distributed execution vs single searcher
   */
  @Override
  public void run() {
    LOG.info("============ Running Sharded Search ============");

    IntStream.range(0, searchers.size()).parallel().forEach(i -> {
      HnswDenseSearcher<K> searcher = searchers.get(i);
      String shardOutputPath = args.output.replaceFirst("\\.txt$", ".shard" + String.format("%02d", i) + ".txt");
      LOG.info("Processing shard {} -> {}", i, shardOutputPath);

      try {
        SortedMap<K, ScoredDoc[]> shardResults = searcher.batch_search(queries, qids, args.hits, threadsPerShard);
        
        try (RunOutputWriter<K> writer = new RunOutputWriter<>(shardOutputPath, args.format, args.runtag, null)) {
          shardResults.forEach((qid, docs) -> {
            try {
              writer.writeTopic(qid, queries.get(qids.indexOf(qid)), docs);
            } catch (JsonProcessingException e) {
              throw new RuntimeException(e);
            }
          });
        }
        
        // Close searcher immediately after use
        searcher.close();
        LOG.info("Closed searcher for shard {}", i);
        
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });

    LOG.info("All shards processed. Results written to individual shard files.");
  }
  /*
   * Error handling and logging approach:
   * - Maintains similar command line argument handling to non-sharded
   * - Adds additional logging specific to shard processing
   * - Uses same exception hierarchy and messaging patterns
   * - Should standardize on logging approach used in non-sharded
   */

  /*
   * Output handling differences:
   * - Non-sharded writes single merged output file
   * - Sharded writes separate file per shard
   * - Uses same output format and runtag
   * - No current support for merging shard results
   */

  /*
   * Future improvements needed:
   * - Move searcher closing to close() method
   * - Implement dynamic thread allocation per shard
   * - Add support for merged output file
   * - Standardize logging approach
   * - Improve memory management
   * - Add proper JavaDoc documentation
   */
  public static void main(String[] args) throws Exception {
    Args searchArgs = new Args();
    CmdLineParser parser = new CmdLineParser(searchArgs, ParserProperties.defaults().withUsageWidth(120));

    try {
      parser.parseArgument(args);
    } catch (CmdLineException e) {
      if (searchArgs.options) {
        System.err.printf("Options for %s:\n\n", SearchShardedHnswDenseVectors.class.getSimpleName());
        parser.printUsage(System.err);

        List<String> required = new ArrayList<>();
        parser.getOptions().forEach((option) -> {
          if (option.option.required()) {
            required.add(option.option.toString());
          }
        });

        System.err.printf("\nRequired options are %s\n", required);
      } else {
        System.err.printf("Error: %s. For help, use \"-options\" to print out information about options.\n",
          e.getMessage());
      }

      return;
    }
    SearchShardedHnswDenseVectors<?> searcher = new SearchShardedHnswDenseVectors<>(searchArgs);
    try {
      searcher.run();
    } catch (RuntimeException e) {
      System.err.printf("Error: %s\n", e.getMessage());
    }
  }
}
