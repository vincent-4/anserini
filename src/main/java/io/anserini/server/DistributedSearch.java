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

package io.anserini.server;

import io.anserini.search.ScoredDoc;
import io.anserini.search.HnswDenseSearcher;
import io.anserini.util.PrebuiltIndexHandler;
import io.anserini.index.IndexInfo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.HashMap;

public class DistributedSearch {
    private final List<String> shardIndexDirs;
    private final String baseIndex;
    private final ExecutorService executorService;
    private final Map<String, Object> settings = new ConcurrentHashMap<>();
    
    private static final int DEFAULT_THREADS_PER_SHARD = 4;

    public DistributedSearch(String baseIndex) {
        this.baseIndex = baseIndex;
        this.shardIndexDirs = initializeShardIndexes(baseIndex);
        this.executorService = Executors.newFixedThreadPool(10 * DEFAULT_THREADS_PER_SHARD);
    }

    private List<String> initializeShardIndexes(String baseIndex) {
        List<String> indexDirs = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            String shardIndex = String.format("%s-shard%02d", baseIndex, i);
            try {
                PrebuiltIndexHandler handler = new PrebuiltIndexHandler(shardIndex);
                handler.initialize();
                handler.download();
                indexDirs.add(handler.decompressIndex());
            } catch (Exception e) {
                throw new RuntimeException("Failed to initialize shard " + i, e);
            }
        }
        return indexDirs;
    }

    public List<Map<String, Object>> search(String query, int hits,
        Integer efSearch, String encoder, String queryGenerator) {
        
        IndexInfo indexInfo = IndexInfo.get(baseIndex);
        HnswDenseSearcher.Args args = new HnswDenseSearcher.Args();
        args.efSearch = efSearch != null ? efSearch : IndexInfo.DEFAULT_EF_SEARCH;
        args.encoder = encoder != null ? encoder.replace(".class", "") 
            : indexInfo.encoder != null ? indexInfo.encoder.replace(".class", "") : null;
        args.queryGenerator = queryGenerator != null ? queryGenerator.replace(".class", "")
            : indexInfo.queryGenerator.replace(".class", "");

        List<Future<List<Map<String, Object>>>> futures = shardIndexDirs.stream()
            .map(indexDir -> executorService.submit(() -> {
                args.index = indexDir;
                try (HnswDenseSearcher<Float> searcher = new HnswDenseSearcher<Float>(args)) {
                    ScoredDoc[] results = searcher.search(query, hits);
                    return convertResults(results);
                }
            }))
            .collect(Collectors.toList());
            
        List<Map<String, Object>> mergedResults = new ArrayList<>();
        for (Future<List<Map<String, Object>>> future : futures) {
            try {
                mergedResults.addAll(future.get());
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        
        return mergedResults.stream()
            .sorted((a, b) -> Double.compare((Double)b.get("score"), (Double)a.get("score")))
            .limit(hits)
            .collect(Collectors.toList());
    }

    private List<Map<String, Object>> convertResults(ScoredDoc[] results) {
        return Arrays.stream(results)
            .map(r -> {
                Map<String, Object> result = new HashMap<>();
                result.put("docid", r.docid);
                result.put("score", r.score);
                return result;
            })
            .collect(Collectors.toList());
    }

    public void setThreadsPerShard(int value) {
        if (value <= 0) throw new IllegalArgumentException("threadsPerShard must be positive");
        settings.put("threadsPerShard", value);
    }

    public Integer getThreadsPerShard() {
        return (Integer) settings.getOrDefault("threadsPerShard", DEFAULT_THREADS_PER_SHARD);
    }
}