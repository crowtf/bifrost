package vectorstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/stdlib"
	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/postgres"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	gormlogger "gorm.io/gorm/logger"
)

const (
	PgVectorTestTimeout = 60 * time.Second
	// PgVectorCleanupTimeout bounds a fixture's namespace drop, which runs on a
	// context of its own: the test's may have expired by then.
	PgVectorCleanupTimeout = 10 * time.Second
	PgVectorTestNamespace  = "bifrost_test_pgvector"
	PgVectorTestDimension  = 384
	// PgVectorTestDefaultDSN is the postgres service in tests/docker-compose.yml.
	PgVectorTestDefaultDSN = "host=localhost port=5432 user=bifrost password=bifrost_password dbname=bifrost sslmode=disable"
)

// pgTestLogStore stands in for the logs store: pgvector only needs its database.
type pgTestLogStore struct{ db *gorm.DB }

func (s pgTestLogStore) ScopedDB(ctx context.Context) *gorm.DB { return s.db.WithContext(ctx) }

type PgVectorTestSetup struct {
	Store  *PgVectorStore
	DB     *gorm.DB
	Logger schemas.Logger
	ctx    context.Context
	cancel context.CancelFunc
}

// openPgVectorTestDB opens the test database named by PGVECTOR_TEST_DSN, in
// keyword=value or URI form. settings are session settings every connection
// starts with.
func openPgVectorTestDB(t *testing.T, settings map[string]string) *gorm.DB {
	t.Helper()
	db, err := openPgVectorDB(getEnvWithDefault("PGVECTOR_TEST_DSN", PgVectorTestDefaultDSN), settings)
	if err != nil {
		t.Fatalf("Failed to open the pgvector test database (set PGVECTOR_TEST_DSN): %v", err)
	}
	sqlDB, err := db.DB()
	require.NoError(t, err)
	t.Cleanup(func() { _ = sqlDB.Close() })
	return db
}

// openPgVectorDB parses dsn with pgx, which reads both DSN forms, and adds
// settings as runtime parameters rather than editing the DSN text.
func openPgVectorDB(dsn string, settings map[string]string) (*gorm.DB, error) {
	config, err := pgx.ParseConfig(dsn)
	if err != nil {
		return nil, err
	}
	for key, value := range settings {
		config.RuntimeParams[key] = value
	}
	sqlDB := stdlib.OpenDB(*config)
	db, err := gorm.Open(postgres.New(postgres.Config{Conn: sqlDB}), &gorm.Config{Logger: gormlogger.Default.LogMode(gormlogger.Silent)})
	if err != nil {
		// gorm.Open pings and returns its error without closing the pool it was given.
		_ = sqlDB.Close()
		return nil, err
	}
	return db, nil
}

func NewPgVectorTestSetup(t *testing.T) *PgVectorTestSetup {
	t.Helper()
	db := openPgVectorTestDB(t, nil)
	logger := bifrost.NewDefaultLogger(schemas.LogLevelInfo)
	ctx, cancel := context.WithTimeout(context.Background(), PgVectorTestTimeout)
	store, err := newPgVectorStore(ctx, &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: db}}, logger)
	if err != nil {
		cancel()
		t.Fatalf("Failed to create pgvector store: %v", err)
	}
	return &PgVectorTestSetup{Store: store, DB: db, Logger: logger, ctx: ctx, cancel: cancel}
}

func (ts *PgVectorTestSetup) Cleanup(t *testing.T) {
	defer ts.cancel()
	assert.NoError(t, ts.dropNamespace(PgVectorTestNamespace))
}

// dropNamespace drops a namespace on a context of its own, so cleanup still
// runs after the test's deadline has passed.
func (ts *PgVectorTestSetup) dropNamespace(name string) error {
	ctx, cancel := context.WithTimeout(context.Background(), PgVectorCleanupTimeout)
	defer cancel()
	return ts.Store.DeleteNamespace(ctx, name)
}

// freshPgNamespace returns a namespace name dropped before and after the test.
func (ts *PgVectorTestSetup) freshPgNamespace(t *testing.T, name string) string {
	t.Helper()
	require.NoError(t, ts.Store.DeleteNamespace(ts.ctx, name))
	t.Cleanup(func() { _ = ts.dropNamespace(name) })
	return name
}

// ============================================================================
// Unit tests: no database
// ============================================================================

func TestPgVectorWhereTranslatesEveryOperator(t *testing.T) {
	tests := []struct {
		name  string
		query Query
		sql   string
		args  []any
	}{
		{
			name:  "equal scalar uses containment for the GIN index",
			query: Query{Field: "type", Operator: QueryOperatorEqual, Value: "chat"},
			sql:   "properties @> ?::jsonb",
			args:  []any{`{"type":"chat"}`},
		},
		{
			name:  "equal list matches nothing, as in chromem",
			query: Query{Field: "tags", Operator: QueryOperatorEqual, Value: []string{"a", "b"}},
			sql:   "FALSE",
		},
		{
			name:  "equal typed nil slice matches nothing, not a null property",
			query: Query{Field: "tags", Operator: QueryOperatorEqual, Value: []string(nil)},
			sql:   "FALSE",
		},
		{
			name:  "not equal list matches every present value, as in chromem",
			query: Query{Field: "tags", Operator: QueryOperatorNotEqual, Value: []string{"a"}},
			sql:   "((properties->E'tags') IS NOT NULL AND (properties->E'tags') <> 'null'::jsonb)",
		},
		{
			name:  "not equal requires the field to be present",
			query: Query{Field: "type", Operator: QueryOperatorNotEqual, Value: "chat"},
			sql:   "((properties->E'type') IS NOT NULL AND (properties->E'type') <> 'null'::jsonb AND (properties->E'type') <> ?::jsonb)",
			args:  []any{`"chat"`},
		},
		{
			name:  "number range compares numbers as jsonb, for the expression index",
			query: Query{Field: "timestamp", Operator: QueryOperatorGreaterThanOrEqual, Value: int64(1700000000)},
			sql:   "(jsonb_typeof((properties->E'timestamp')) = 'number' AND (properties->E'timestamp') >= ?::jsonb)",
			args:  []any{`1700000000`},
		},
		{
			name:  "string range compares bytes, not the database collation",
			query: Query{Field: "model", Operator: QueryOperatorGreaterThan, Value: "Z"},
			sql:   `(jsonb_typeof((properties->E'model')) = 'string' AND (properties->>E'model') COLLATE "C" > ?)`,
			args:  []any{"Z"},
		},
		{
			name:  "like without wildcards is a substring match",
			query: Query{Field: "key", Operator: QueryOperatorLike, Value: "a_b"},
			sql:   "(jsonb_typeof((properties->E'key')) = 'string' AND properties->>E'key' LIKE ?)",
			args:  []any{`%a\_b%`},
		},
		{
			name:  "contains any matches an array element or the scalar",
			query: Query{Field: "tier", Operator: QueryOperatorContainsAny, Value: []string{"simple", "complex"}},
			sql:   "((properties @> ?::jsonb OR properties @> ?::jsonb) OR (properties @> ?::jsonb OR properties @> ?::jsonb))",
			args:  []any{`{"tier":["simple"]}`, `{"tier":"simple"}`, `{"tier":["complex"]}`, `{"tier":"complex"}`},
		},
		{
			name:  "contains all needs every value",
			query: Query{Field: "tier", Operator: QueryOperatorContainsAll, Value: []string{"a", "b"}},
			sql:   "((properties @> ?::jsonb OR properties @> ?::jsonb) AND (properties @> ?::jsonb OR properties @> ?::jsonb))",
			args:  []any{`{"tier":["a"]}`, `{"tier":"a"}`, `{"tier":["b"]}`, `{"tier":"b"}`},
		},
		{
			name:  "a nil element matches nothing, not a null property",
			query: Query{Field: "tags", Operator: QueryOperatorContainsAny, Value: []any{nil, "a"}},
			sql:   "(FALSE OR (properties @> ?::jsonb OR properties @> ?::jsonb))",
			args:  []any{`{"tags":["a"]}`, `{"tags":"a"}`},
		},
		{
			name:  "a map element matches nothing, not a stored object by subset",
			query: Query{Field: "tags", Operator: QueryOperatorContainsAny, Value: []any{map[string]any{"k": "v"}, "a"}},
			sql:   "(FALSE OR (properties @> ?::jsonb OR properties @> ?::jsonb))",
			args:  []any{`{"tags":["a"]}`, `{"tags":"a"}`},
		},
		{
			name:  "contains nothing matches nothing",
			query: Query{Field: "tier", Operator: QueryOperatorContainsAny, Value: []string{}},
			sql:   "FALSE",
		},
		{
			name:  "is null covers a missing field and a JSON null",
			query: Query{Field: "parent", Operator: QueryOperatorIsNull},
			sql:   "((properties->E'parent') IS NULL OR (properties->E'parent') = 'null'::jsonb)",
		},
		{
			name:  "a quote in a field name is escaped, not executed",
			query: Query{Field: "it's", Operator: QueryOperatorIsNotNull},
			sql:   "((properties->E'it''s') IS NOT NULL AND (properties->E'it''s') <> 'null'::jsonb)",
		},
		{
			name:  "a backslash is escaped whatever standard_conforming_strings is",
			query: Query{Field: `a\b`, Operator: QueryOperatorIsNotNull},
			sql:   `((properties->E'a\\b') IS NOT NULL AND (properties->E'a\\b') <> 'null'::jsonb)`,
		},
		{
			name:  "a question mark is hex-escaped so GORM does not take it for a placeholder",
			query: Query{Field: "tag?", Operator: QueryOperatorIsNotNull},
			sql:   `((properties->E'tag\x3f') IS NOT NULL AND (properties->E'tag\x3f') <> 'null'::jsonb)`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conditions, args, err := pgWhere([]Query{tt.query})
			require.NoError(t, err)
			require.Len(t, conditions, 1)
			assert.Equal(t, tt.sql, conditions[0])
			assert.Equal(t, tt.args, args)
		})
	}
}

func TestPgVectorWhereRejectsWhatItCannotTranslate(t *testing.T) {
	for name, query := range map[string]Query{
		"range over a boolean":  {Field: "warp_log", Operator: QueryOperatorGreaterThan, Value: true},
		"like over a number":    {Field: "size", Operator: QueryOperatorLike, Value: 5},
		"unknown operator":      {Field: "size", Operator: "Between", Value: 5},
		"unknown operator, nil": {Field: "size", Operator: "Between"},
		"empty field":           {Field: "", Operator: QueryOperatorEqual, Value: 5},
		"NUL in a field":        {Field: "a\x00b", Operator: QueryOperatorEqual, Value: 5},
	} {
		t.Run(name, func(t *testing.T) {
			_, _, err := pgWhere([]Query{query})
			assert.ErrorIs(t, err, ErrQuerySyntax)
		})
	}
}

func TestPgVectorLikePattern(t *testing.T) {
	assert.Equal(t, "%abc%", pgLikePattern("abc"))
	assert.Equal(t, "abc%", pgLikePattern("abc*"))
	assert.Equal(t, "%abc", pgLikePattern("%abc"))
	assert.Equal(t, `%a\\b%`, pgLikePattern(`a\b`))
}

func TestPgVectorVectorTextRoundTrip(t *testing.T) {
	vector := []float32{0.5, -1.25, 3e-7, 0}
	text := formatPgVector(vector)
	assert.Equal(t, "[0.5,-1.25,3e-07,0]", text)
	parsed, err := parsePgVector(text)
	require.NoError(t, err)
	assert.Equal(t, vector, parsed)

	_, err = parsePgVector("0.5,1")
	assert.Error(t, err)
}

func TestPgVectorRequiresIterativeScans(t *testing.T) {
	assert.Equal(t, "0.8.0", pgMinVersion, "0.7 has no iterative scans, so a selective filter can miss rows that match")
	assert.Equal(t, -1, compareVersions("0.7.4", pgMinVersion))
}

func TestPgVectorCompareVersions(t *testing.T) {
	assert.Equal(t, 0, compareVersions("0.8", "0.8.0"))
	assert.Equal(t, -1, compareVersions("0.7.4", "0.8.0"))
	assert.Equal(t, 1, compareVersions("0.10.0", "0.8.0"), "compared numerically, not as text")
}

func TestPgVectorTableNameFitsAnIdentifier(t *testing.T) {
	long := strings.Repeat("complexity_capture_router_", 10)
	name := pgTableName(long, pgMaxDimension)
	assert.LessOrEqual(t, len(name)+len("_hnsw"), 63, "index names derive from the table name and must fit too")
	assert.Equal(t, name, pgTableName(long, pgMaxDimension), "deterministic")
	assert.NotEqual(t, pgTableName("a", 768), pgTableName("a", 1536), "a new dimension is a new table")
}

// TestPgVectorIndexFollowsTheDimension pins which HNSW index a namespace gets,
// and that the search expression is the one that index is built on.
func TestPgVectorIndexFollowsTheDimension(t *testing.T) {
	hnsw := func(dimension int) string {
		for _, statement := range pgCreateNamespaceSQL("t", dimension, nil) {
			if strings.Contains(statement, "USING hnsw") {
				return statement
			}
		}
		return ""
	}
	assert.Contains(t, hnsw(1536), "USING hnsw (embedding vector_cosine_ops)")
	assert.Equal(t, "(embedding <=> ?::vector)", pgDistanceSQL(1536))

	assert.Contains(t, hnsw(3072), "USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)")
	assert.Equal(t, "(embedding::halfvec(3072) <=> ?::halfvec(3072))", pgDistanceSQL(3072))

	assert.Empty(t, hnsw(5000), "above the halfvec limit a namespace has no HNSW index")
}

func TestPgVectorIndexesIntegerProperties(t *testing.T) {
	statements := pgCreateNamespaceSQL("t", 3, map[string]VectorStoreProperties{
		"timestamp": {DataType: VectorStorePropertyTypeInteger},
		"model":     {DataType: VectorStorePropertyTypeString},
	})
	var btree []string
	for _, statement := range statements {
		if strings.Contains(statement, "((properties->") {
			btree = append(btree, statement)
		}
	}
	require.Len(t, btree, 1, "string properties are served by the GIN index")
	assert.Contains(t, btree[0], "((properties->E'timestamp'))")
}

func TestPgVectorRefusesWithoutAPostgresLogStore(t *testing.T) {
	logger := bifrost.NewDefaultLogger(schemas.LogLevelInfo)

	_, err := newPgVectorStore(context.Background(), &PgVectorConfig{}, Dependencies{}, logger)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no logs store is configured")

	db, err := gorm.Open(sqlite.Open("file::memory:"), &gorm.Config{Logger: gormlogger.Default.LogMode(gormlogger.Silent)})
	require.NoError(t, err)
	_, err = newPgVectorStore(context.Background(), &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: db}}, logger)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "the logs store is sqlite")
}

func TestPgVectorConfigUnmarshal(t *testing.T) {
	for _, raw := range []string{`{"enabled":true,"type":"pgvector"}`, `{"enabled":true,"type":"pgvector","config":{}}`} {
		var config Config
		require.NoError(t, config.UnmarshalJSON([]byte(raw)))
		assert.Equal(t, VectorStoreTypePgVector, config.Type)
		assert.IsType(t, PgVectorConfig{}, config.Config)
	}
	// A config built in code, with no settings, reads back after marshaling.
	marshaled, err := json.Marshal(Config{Enabled: true, Type: VectorStoreTypePgVector})
	require.NoError(t, err)
	assert.JSONEq(t, `{"enabled":true,"type":"pgvector","config":{}}`, string(marshaled))
	var roundTripped Config
	require.NoError(t, json.Unmarshal(marshaled, &roundTripped))
	assert.Equal(t, PgVectorConfig{}, roundTripped.Config)

	// The schema allows only an omitted config or {}: anything else would be a
	// setting Bifrost silently ignores.
	for name, raw := range map[string]string{
		"null":          `{"enabled":true,"type":"pgvector","config":null}`,
		"a setting":     `{"enabled":true,"type":"pgvector","config":{"addr":"localhost:6379"}}`,
		"not an object": `{"enabled":true,"type":"pgvector","config":"x"}`,
	} {
		t.Run(name, func(t *testing.T) {
			var config Config
			assert.Error(t, config.UnmarshalJSON([]byte(raw)))
		})
	}
}

// ============================================================================
// Integration tests: need Postgres with pgvector (tests/docker-compose.yml)
// ============================================================================

// TestPgVectorRefusesTypesOffTheSearchPath pins that an extension installed in
// a schema the connection cannot see fails at startup, naming the schema,
// rather than on the first CREATE TABLE that names the vector type.
func TestPgVectorRefusesTypesOffTheSearchPath(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	// The normal fixture installs the extension on a fresh database first.
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	var schema string
	require.NoError(t, setup.DB.Raw("SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'vector'").Scan(&schema).Error)
	require.NotEmpty(t, schema)

	db := openPgVectorTestDB(t, map[string]string{"search_path": "pg_catalog"})
	_, err := newPgVectorStore(context.Background(), &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: db}}, bifrost.NewDefaultLogger(schemas.LogLevelError))
	require.Error(t, err)
	assert.Contains(t, err.Error(), fmt.Sprintf("installed in schema %q, which is not on the search_path", schema))
	assert.Contains(t, err.Error(), fmt.Sprintf("SET search_path = pg_catalog, %q", schema), "the suggested command keeps the current entries")
}

func TestPgVectorStore_InterfaceCompliance(t *testing.T) {
	var _ VectorStore = (*PgVectorStore)(nil)
	var _ ServerFilteredReader = (*PgVectorStore)(nil)
}

func TestVectorStoreFactory_PgVector(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	db := openPgVectorTestDB(t, nil)
	logger := bifrost.NewDefaultLogger(schemas.LogLevelInfo)
	config := &Config{Enabled: true, Type: VectorStoreTypePgVector}

	_, err := NewVectorStore(context.Background(), config, logger)
	require.Error(t, err, "pgvector cannot be built without the logs store")

	store, err := NewVectorStoreWithDependencies(context.Background(), config, Dependencies{LogStore: pgTestLogStore{db: db}}, logger)
	require.NoError(t, err)
	assert.IsType(t, &PgVectorStore{}, store)
	assert.NoError(t, store.Ping(context.Background()))
}

func TestPgVectorStore_CreateNamespaceRejectsDimensionChange(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgDimGuard")
	props := map[string]VectorStoreProperties{"type": {DataType: VectorStorePropertyTypeString}}

	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 512, props))
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 512, props))

	err := setup.Store.CreateNamespace(setup.ctx, namespace, 1024, props)
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrDimensionMismatch)
	assert.Contains(t, err.Error(), "already exists with dimension 512 but config requires 1024")
}

// TestPgVectorStore_CreateNamespaceIsSafeAcrossReplicas pins the advisory
// lock: replicas starting together all create the same namespace.
func TestPgVectorStore_CreateNamespaceIsSafeAcrossReplicas(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgConcurrentCreate")
	props := map[string]VectorStoreProperties{"timestamp": {DataType: VectorStorePropertyTypeInteger}}

	const replicas = 8
	errs := make([]error, replicas)
	var wg sync.WaitGroup
	for i := range replicas {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// A store per replica, so none shares another's namespace cache.
			store, err := newPgVectorStore(setup.ctx, &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: setup.DB}}, setup.Logger)
			if err != nil {
				errs[i] = err
				return
			}
			errs[i] = store.CreateNamespace(setup.ctx, namespace, 3, props)
		}()
	}
	wg.Wait()
	for _, err := range errs {
		assert.NoError(t, err)
	}
	names, err := setup.Store.ListNamespaces(setup.ctx, namespace)
	require.NoError(t, err)
	assert.Equal(t, []string{namespace}, names)
}

func TestPgVectorStore_GetNearestScoresCosineSimilarity(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgCosineScore")
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 2, map[string]VectorStoreProperties{"label": {DataType: VectorStorePropertyTypeString}}))

	require.NoError(t, setup.Store.Add(setup.ctx, namespace, generateUUID(), unitVectorAtCosine(2, 0.9), map[string]interface{}{"label": "close"}))
	require.NoError(t, setup.Store.Add(setup.ctx, namespace, generateUUID(), unitVectorAtCosine(2, 0.7), map[string]interface{}{"label": "far"}))

	results, err := setup.Store.GetNearest(setup.ctx, namespace, unitVectorAtCosine(2, 1), nil, []string{"label"}, 0.8, 10)
	require.NoError(t, err)
	require.Len(t, results, 1, "only the vector at cosine 0.9 clears a 0.8 threshold")
	assert.Equal(t, "close", results[0].Properties["label"])
	require.NotNil(t, results[0].Score)
	assert.InDelta(t, 0.9, *results[0].Score, 1e-3)
}

// TestPgVectorStore_GetNearestFillsTheLimitPastAFilter pins iterative scans: a
// filter that only distant rows pass still yields limit rows, where a plain
// HNSW scan stops after ef_search candidates and drops the filtered ones.
func TestPgVectorStore_GetNearestFillsTheLimitPastAFilter(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgIterativeScan")
	const dimension = 8
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, dimension, map[string]VectorStoreProperties{"kind": {DataType: VectorStorePropertyTypeString}}))
	// A table this small is cheaper to scan exactly, which would hide the HNSW
	// behaviour under test, so this store's sessions may not seq scan.
	indexOnly, err := newPgVectorStore(setup.ctx, &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: openPgVectorTestDB(t, map[string]string{"enable_seqscan": "off"})}}, setup.Logger)
	require.NoError(t, err)

	query := unitVectorAtCosine(dimension, 1)
	// 300 near rows the filter rejects, then 20 far rows it accepts.
	for i := range 300 {
		require.NoError(t, setup.Store.Add(setup.ctx, namespace, fmt.Sprintf("near-%03d", i), unitVectorAtCosine(dimension, 0.99-float64(i)*0.0001), map[string]interface{}{"kind": "other"}))
	}
	for i := range 20 {
		require.NoError(t, setup.Store.Add(setup.ctx, namespace, fmt.Sprintf("far-%02d", i), unitVectorAtCosine(dimension, 0.5-float64(i)*0.001), map[string]interface{}{"kind": "wanted"}))
	}

	// NotEqual, because the GIN index can serve Equal, and the planner then
	// reads the matching rows and sorts them exactly instead of using HNSW.
	results, err := indexOnly.GetNearest(setup.ctx, namespace, query, []Query{{Field: "kind", Operator: QueryOperatorNotEqual, Value: "other"}}, nil, 0, 10)
	require.NoError(t, err)
	require.Len(t, results, 10)
	for i := 1; i < len(results); i++ {
		assert.GreaterOrEqual(t, *results[i-1].Score, *results[i].Score, "results stay ordered by similarity")
	}
	assert.Equal(t, "far-00", results[0].ID)

	// A limit above hnsw.ef_search's default of 40 is honoured too.
	all, err := indexOnly.GetNearest(setup.ctx, namespace, query, nil, nil, 0, 100)
	require.NoError(t, err)
	assert.Len(t, all, 100)
}

// TestPgVectorStore_SearchUsesTheIndex checks the planner can serve a search
// from the HNSW index, for vector and for halfvec. If the ORDER BY expression
// drifts from the index expression, every search silently becomes exact.
func TestPgVectorStore_SearchUsesTheIndex(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	for _, dimension := range []int{16, 3072} {
		t.Run(fmt.Sprint(dimension), func(t *testing.T) {
			namespace := setup.freshPgNamespace(t, fmt.Sprintf("TestPgIndexUse%d", dimension))
			require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, dimension, nil))
			require.NoError(t, setup.Store.Add(setup.ctx, namespace, "a", unitVectorAtCosine(dimension, 0.5), nil))
			ns, err := setup.Store.namespace(setup.ctx, namespace)
			require.NoError(t, err)

			var plan []string
			err = setup.DB.Transaction(func(tx *gorm.DB) error {
				if err := tx.Exec("SET LOCAL enable_seqscan = off").Error; err != nil {
					return err
				}
				return tx.Raw("EXPLAIN SELECT id, "+pgDistanceSQL(dimension)+" AS distance FROM "+ns.table+" WHERE embedding IS NOT NULL ORDER BY distance LIMIT 5",
					formatPgVector(unitVectorAtCosine(dimension, 1))).Scan(&plan).Error
			})
			require.NoError(t, err)
			assert.Contains(t, strings.Join(plan, "\n"), ns.table+"_hnsw", "plan:\n%s", strings.Join(plan, "\n"))

			results, err := setup.Store.GetNearest(setup.ctx, namespace, unitVectorAtCosine(dimension, 1), nil, nil, 0, 5)
			require.NoError(t, err)
			require.Len(t, results, 1)
			assert.InDelta(t, 0.5, *results[0].Score, 1e-3, "halfvec keeps enough precision for the score")
		})
	}
}

// TestPgVectorStore_QueriesMatchChromem runs every operator against pgvector
// and chromem over the same entries. Chromem evaluates queries in Go and is the
// reference; the SQL translation must select the same entries.
func TestPgVectorStore_QueriesMatchChromem(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	reference := NewChromemTestSetup(t)
	defer reference.Cleanup(t)

	const dimension = 4
	namespace := setup.freshPgNamespace(t, "TestPgParity")
	props := map[string]VectorStoreProperties{
		"timestamp": {DataType: VectorStorePropertyTypeInteger},
		"model":     {DataType: VectorStorePropertyTypeString},
		"warp_log":  {DataType: VectorStorePropertyTypeBoolean},
		"tags":      {DataType: VectorStorePropertyTypeStringArray},
	}
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, dimension, props))
	require.NoError(t, reference.Store.CreateNamespace(reference.ctx, namespace, dimension, props))
	defer func() { _ = reference.Store.DeleteNamespace(reference.ctx, namespace) }()

	entries := map[string]map[string]interface{}{
		"e1": {"timestamp": int64(100), "model": "gpt-4o", "warp_log": true, "tags": []string{"a", "b"}},
		"e2": {"timestamp": int64(200), "model": "gpt-4o-mini", "warp_log": false, "tags": []string{"b"}},
		"e3": {"timestamp": int64(300), "model": "claude", "tags": []string{"c"}, "parent": nil},
		"e4": {"timestamp": 250.5, "model": "o3", "warp_log": true},
		"e5": {"timestamp": "late", "model": "gpt_5", "tags": "a"},
		"e6": {},
		"e7": {"tags": nil, "tag?": "yes"},
		"e8": {"tags": []any{map[string]any{"k": "v", "extra": 1}}},
	}
	for id, metadata := range entries {
		vector := []float32{1, float32(len(id)), 0.5, 0.25}
		require.NoError(t, setup.Store.Add(setup.ctx, namespace, id, vector, metadata))
		require.NoError(t, reference.Store.Add(reference.ctx, namespace, id, vector, metadata))
	}

	queries := map[string][]Query{
		"equal string":         {{Field: "model", Operator: QueryOperatorEqual, Value: "gpt-4o"}},
		"equal bool":           {{Field: "warp_log", Operator: QueryOperatorEqual, Value: true}},
		"equal int to float":   {{Field: "timestamp", Operator: QueryOperatorEqual, Value: 200.0}},
		"not equal":            {{Field: "warp_log", Operator: QueryOperatorNotEqual, Value: true}},
		"greater or equal":     {{Field: "timestamp", Operator: QueryOperatorGreaterThanOrEqual, Value: int64(200)}},
		"less than":            {{Field: "timestamp", Operator: QueryOperatorLessThan, Value: 250}},
		"range window":         {{Field: "timestamp", Operator: QueryOperatorGreaterThanOrEqual, Value: 150}, {Field: "timestamp", Operator: QueryOperatorLessThan, Value: 300}},
		"string range":         {{Field: "model", Operator: QueryOperatorGreaterThan, Value: "gpt-4o"}},
		"string range by byte": {{Field: "model", Operator: QueryOperatorGreaterThan, Value: "Z"}},
		"like on a key with ?": {{Field: "tag?", Operator: QueryOperatorLike, Value: "ye"}},
		"contains a nil":       {{Field: "tags", Operator: QueryOperatorContainsAny, Value: []any{nil, "c"}}},
		"contains all and nil": {{Field: "tags", Operator: QueryOperatorContainsAll, Value: []any{"b", nil}}},
		"equal list":           {{Field: "tags", Operator: QueryOperatorEqual, Value: []string{"a", "b"}}},
		"contains a map":       {{Field: "tags", Operator: QueryOperatorContainsAny, Value: []any{map[string]any{"k": "v"}}}},
		"not equal list":       {{Field: "tags", Operator: QueryOperatorNotEqual, Value: []string{"a", "b"}}},
		"like substring":       {{Field: "model", Operator: QueryOperatorLike, Value: "4o"}},
		"like prefix":          {{Field: "model", Operator: QueryOperatorLike, Value: "gpt*"}},
		"like literal _":       {{Field: "model", Operator: QueryOperatorLike, Value: "gpt_"}},
		"contains any":         {{Field: "tags", Operator: QueryOperatorContainsAny, Value: []string{"a", "c"}}},
		"contains all":         {{Field: "tags", Operator: QueryOperatorContainsAll, Value: []string{"a", "b"}}},
		"contains any scalar":  {{Field: "tags", Operator: QueryOperatorContainsAny, Value: "a"}},
		"is null":              {{Field: "parent", Operator: QueryOperatorIsNull}},
		"is not null":          {{Field: "warp_log", Operator: QueryOperatorIsNotNull}},
		"missing field equals": {{Field: "nope", Operator: QueryOperatorEqual, Value: "x"}},
		"warp retention":       {{Field: "warp_log", Operator: QueryOperatorEqual, Value: true}, {Field: "timestamp", Operator: QueryOperatorLessThan, Value: int64(200)}},
	}
	ids := func(results []SearchResult) []string {
		out := make([]string, 0, len(results))
		for _, result := range results {
			out = append(out, result.ID)
		}
		sort.Strings(out)
		return out
	}
	for name, filter := range queries {
		t.Run(name, func(t *testing.T) {
			want, _, err := reference.Store.GetAll(reference.ctx, namespace, filter, nil, nil, 100)
			require.NoError(t, err)
			got, _, err := setup.Store.GetAll(setup.ctx, namespace, filter, nil, nil, 100)
			require.NoError(t, err)
			if name != "missing field equals" && name != "contains all and nil" && name != "equal list" && name != "contains a map" {
				require.NotEmpty(t, want, "a query that matches nothing on either store proves nothing")
			}
			assert.Equal(t, ids(want), ids(got))
		})
	}
}

// TestPgVectorStore_PropertiesRoundTripTheirTypes pins that properties come
// back as the types the interface declares, as they do from chromem.
func TestPgVectorStore_PropertiesRoundTripTheirTypes(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgTypes")
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 2, nil))
	require.NoError(t, setup.Store.Add(setup.ctx, namespace, "x", []float32{1, 0}, map[string]interface{}{
		"count": int64(7), "ratio": 0.5, "ok": true, "name": "n", "tags": []string{"a"},
	}))
	result, err := setup.Store.GetChunk(setup.ctx, namespace, "x")
	require.NoError(t, err)
	assert.Equal(t, map[string]interface{}{"count": int64(7), "ratio": 0.5, "ok": true, "name": "n", "tags": []string{"a"}}, result.Properties)
}

// TestPgVectorStore_StoresEntriesWithoutVectors pins RequiresVectors() false:
// an entry with no embedding is stored and read, and searches pass over it.
func TestPgVectorStore_StoresEntriesWithoutVectors(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgNoVector")
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 2, nil))

	assert.False(t, setup.Store.RequiresVectors())
	require.NoError(t, setup.Store.Add(setup.ctx, namespace, "meta", nil, map[string]interface{}{"k": "v"}))
	require.NoError(t, setup.Store.Add(setup.ctx, namespace, "vec", []float32{1, 0}, nil))

	result, err := setup.Store.GetChunk(setup.ctx, namespace, "meta")
	require.NoError(t, err)
	assert.Equal(t, "v", result.Properties["k"])

	results, err := setup.Store.GetNearest(setup.ctx, namespace, []float32{1, 0}, nil, nil, 0, 10)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, "vec", results[0].ID)
}

func TestPgVectorStore_MissingNamespace(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgMissing")

	_, err := setup.Store.GetChunk(setup.ctx, namespace, "x")
	assert.ErrorIs(t, err, ErrNotFound)
	_, _, err = setup.Store.GetAll(setup.ctx, namespace, nil, nil, nil, 10)
	assert.ErrorIs(t, err, ErrNotFound)
	_, err = setup.Store.GetNearest(setup.ctx, namespace, []float32{1}, nil, nil, 0, 1)
	assert.ErrorIs(t, err, ErrNotFound)
	_, err = setup.Store.DeleteAll(setup.ctx, namespace, nil)
	assert.ErrorIs(t, err, ErrNotFound)
	assert.ErrorIs(t, setup.Store.Add(setup.ctx, namespace, "x", []float32{1}, nil), ErrNotFound)

	assert.NoError(t, setup.Store.Delete(setup.ctx, namespace, "x"), "deleting from a missing namespace is a no-op")
	assert.NoError(t, setup.Store.DeleteNamespace(setup.ctx, namespace), "and so is dropping it")
}

// TestPgVectorStore_NamespaceDroppedByAnotherReplica pins the cache: a replica
// that cached a namespace another replica dropped reports ErrNotFound rather
// than a SQL error, and can create it again.
func TestPgVectorStore_NamespaceDroppedByAnotherReplica(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgDroppedElsewhere")
	other, err := newPgVectorStore(setup.ctx, &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: setup.DB}}, setup.Logger)
	require.NoError(t, err)

	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 2, nil))
	require.NoError(t, setup.Store.Add(setup.ctx, namespace, "x", []float32{1, 0}, nil))
	require.NoError(t, other.DeleteNamespace(setup.ctx, namespace))

	_, err = setup.Store.GetChunk(setup.ctx, namespace, "x")
	assert.True(t, errors.Is(err, ErrNotFound), "got %v", err)
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 2, nil))
	require.NoError(t, setup.Store.Add(setup.ctx, namespace, "x", []float32{1, 0}, nil))
}

// TestPgVectorStore_NamespaceRecreatedByAnotherReplica pins the cache against
// a namespace another replica dropped and recreated at another dimension:
// writes, searches and deletes at the new dimension reach the new table.
func TestPgVectorStore_NamespaceRecreatedByAnotherReplica(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgRecreatedElsewhere")
	other, err := newPgVectorStore(setup.ctx, &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: setup.DB}}, setup.Logger)
	require.NoError(t, err)

	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 2, nil))
	require.NoError(t, setup.Store.Add(setup.ctx, namespace, "old", []float32{1, 0}, nil))
	require.NoError(t, other.DeleteNamespace(setup.ctx, namespace))
	require.NoError(t, other.CreateNamespace(setup.ctx, namespace, 3, nil))
	require.NoError(t, other.Add(setup.ctx, namespace, "kept", []float32{1, 0, 0}, nil))

	require.NoError(t, setup.Store.Add(setup.ctx, namespace, "new", []float32{0, 1, 0}, nil), "a vector at the new dimension is accepted")
	results, err := setup.Store.GetNearest(setup.ctx, namespace, []float32{1, 0, 0}, nil, nil, 0, 10)
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, "kept", results[0].ID)

	// Delete through a cache still naming the dropped table.
	stale, err := newPgVectorStore(setup.ctx, &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: setup.DB}}, setup.Logger)
	require.NoError(t, err)
	require.NoError(t, stale.CreateNamespace(setup.ctx, namespace, 3, nil))
	require.NoError(t, other.DeleteNamespace(setup.ctx, namespace))
	require.NoError(t, other.CreateNamespace(setup.ctx, namespace, 4, nil))
	require.NoError(t, other.Add(setup.ctx, namespace, "gone", []float32{1, 0, 0, 0}, nil))
	require.NoError(t, stale.Delete(setup.ctx, namespace, "gone"))
	_, err = other.GetChunk(setup.ctx, namespace, "gone")
	assert.ErrorIs(t, err, ErrNotFound, "the delete reached the recreated table")

	// Reads and DeleteAll through a cache still naming the dropped table find
	// the row in the recreated one. Each gets a fresh stale cache, since the
	// first call through one corrects it.
	dimension := 4
	staleAfterRecreate := func(t *testing.T) *PgVectorStore {
		stale, err := newPgVectorStore(setup.ctx, &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: setup.DB}}, setup.Logger)
		require.NoError(t, err)
		require.NoError(t, stale.CreateNamespace(setup.ctx, namespace, dimension, nil))
		require.NoError(t, other.DeleteNamespace(setup.ctx, namespace))
		dimension++
		require.NoError(t, other.CreateNamespace(setup.ctx, namespace, dimension, nil))
		vector := make([]float32, dimension)
		vector[0] = 1
		require.NoError(t, other.Add(setup.ctx, namespace, "row", vector, map[string]interface{}{"k": "v"}))
		return stale
	}
	t.Run("GetChunk", func(t *testing.T) {
		result, err := staleAfterRecreate(t).GetChunk(setup.ctx, namespace, "row")
		require.NoError(t, err)
		assert.Equal(t, "v", result.Properties["k"])
	})
	t.Run("GetAll", func(t *testing.T) {
		page, _, err := staleAfterRecreate(t).GetAll(setup.ctx, namespace, nil, nil, nil, 10)
		require.NoError(t, err)
		require.Len(t, page, 1)
		assert.Equal(t, "row", page[0].ID)
	})
	t.Run("DeleteAll", func(t *testing.T) {
		deleted, err := staleAfterRecreate(t).DeleteAll(setup.ctx, namespace, nil)
		require.NoError(t, err)
		require.Len(t, deleted, 1)
		_, err = other.GetChunk(setup.ctx, namespace, "row")
		assert.ErrorIs(t, err, ErrNotFound)
	})
}

// TestPgVectorStore_RetriesANamespaceRecreatedAtTheSameDimension pins the
// retry when another replica drops and recreates a namespace at the same
// dimension between a failed statement and the registry read: the table name
// is the same, and the retry must still run against it.
func TestPgVectorStore_RetriesANamespaceRecreatedAtTheSameDimension(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgRecreatedSameDimension")
	other, err := newPgVectorStore(setup.ctx, &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: setup.DB}}, setup.Logger)
	require.NoError(t, err)
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 2, nil))
	require.NoError(t, other.DeleteNamespace(setup.ctx, namespace))

	calls := 0
	err = setup.Store.withNamespace(setup.ctx, namespace, 0, func(ns pgNamespace) error {
		calls++
		if calls == 1 {
			// The statement fails on the dropped table; the namespace is then
			// recreated before the registry is read again.
			failed := setup.DB.Exec("DELETE FROM " + ns.table).Error
			require.NoError(t, other.CreateNamespace(setup.ctx, namespace, 2, nil))
			return failed
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 2, calls, "the retry ran against the recreated table of the same name")
}

// TestPgVectorStore_CreateNamespaceLeavesAnExistingTableAlone pins that a
// property added to a populated namespace is not indexed at startup, since a
// CREATE INDEX would lock the table against writes, while a table dropped by
// hand is recreated with every index.
func TestPgVectorStore_CreateNamespaceLeavesAnExistingTableAlone(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgAddedProperty")
	table := pgTableName(namespace, 2)
	propertyIndexes := func() int {
		var count int
		require.NoError(t, setup.DB.Raw("SELECT count(*) FROM pg_indexes WHERE tablename = ? AND indexname ~ ?", table, "^"+table+"_p[0-9a-f]{8}$").Scan(&count).Error)
		return count
	}
	timestamp := map[string]VectorStoreProperties{"timestamp": {DataType: VectorStorePropertyTypeInteger}}

	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 2, nil))
	require.NoError(t, setup.Store.Add(setup.ctx, namespace, "x", []float32{1, 0}, map[string]interface{}{"timestamp": int64(1)}))
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 2, timestamp))
	assert.Equal(t, 0, propertyIndexes(), "the existing table gets no new index")

	require.NoError(t, setup.DB.Exec("DROP TABLE "+table).Error)
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, 2, timestamp))
	assert.Equal(t, 1, propertyIndexes(), "a table dropped by hand is rebuilt with its indexes")
}

// TestPgVectorStore_GetNearestSkipsUndefinedScores pins cosine on zero
// vectors: pgvector gives them a NaN distance, which no threshold may accept.
// The namespace is above the halfvec limit so the search is exact; an HNSW
// cosine index does not hold zero vectors at all.
func TestPgVectorStore_GetNearestSkipsUndefinedScores(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	namespace := setup.freshPgNamespace(t, "TestPgZeroVector")
	const dimension = 5000
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, namespace, dimension, nil))
	require.NoError(t, setup.Store.Add(setup.ctx, namespace, "zero", make([]float32, dimension), nil))
	require.NoError(t, setup.Store.Add(setup.ctx, namespace, "close", unitVectorAtCosine(dimension, 0.9), nil))

	results, err := setup.Store.GetNearest(setup.ctx, namespace, unitVectorAtCosine(dimension, 1), nil, nil, 0.8, 10)
	require.NoError(t, err)
	require.Len(t, results, 1, "the zero vector's NaN score does not clear the threshold")
	assert.Equal(t, "close", results[0].ID)
	assert.InDelta(t, 0.9, *results[0].Score, 1e-3)

	_, err = setup.Store.GetNearest(setup.ctx, namespace, make([]float32, dimension), nil, nil, 0, 10)
	assert.ErrorContains(t, err, "zero norm", "a zero query vector has no cosine similarity to anything")
}

func TestPgVectorStore_ListNamespacesByPrefixInByteOrder(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.cancel()
	names := []string{"TestPgList_b", "TestPgList_B", "TestPgList_a", "TestPgListOther"}
	for _, name := range names {
		setup.freshPgNamespace(t, name)
		require.NoError(t, setup.Store.CreateNamespace(setup.ctx, name, 2, nil))
	}
	listed, err := setup.Store.ListNamespaces(setup.ctx, "TestPgList_")
	require.NoError(t, err)
	assert.Equal(t, []string{"TestPgList_B", "TestPgList_a", "TestPgList_b"}, listed)

	none, err := setup.Store.ListNamespaces(setup.ctx, "TestPgNothingHere_")
	require.NoError(t, err)
	assert.NotNil(t, none)
	assert.Empty(t, none)
}

func TestPgVectorStore_DeleteMissingIDSucceeds(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	setup := NewPgVectorTestSetup(t)
	defer setup.Cleanup(t)
	require.NoError(t, setup.Store.CreateNamespace(setup.ctx, PgVectorTestNamespace, PgVectorTestDimension, nil))
	assert.NoError(t, setup.Store.Delete(setup.ctx, PgVectorTestNamespace, generateUUID()))
}

// ============================================================================
// Benchmarks: GetNearest and Add are on the semantic cache's request path
// ============================================================================

// newPgVectorBenchStore seeds a namespace of entries shaped like semantic
// cache entries and returns a store over it.
func newPgVectorBenchStore(b *testing.B, entries int) (*PgVectorStore, string) {
	b.Helper()
	db, err := openPgVectorDB(getEnvWithDefault("PGVECTOR_TEST_DSN", PgVectorTestDefaultDSN), nil)
	if err != nil {
		b.Skipf("pgvector test database unavailable: %v", err)
	}
	// Registered first so it runs last, after the namespace drop below.
	sqlDB, err := db.DB()
	require.NoError(b, err)
	b.Cleanup(func() { _ = sqlDB.Close() })
	ctx := context.Background()
	store, err := newPgVectorStore(ctx, &PgVectorConfig{}, Dependencies{LogStore: pgTestLogStore{db: db}}, bifrost.NewDefaultLogger(schemas.LogLevelError))
	if err != nil {
		b.Skipf("pgvector unavailable: %v", err)
	}
	namespace := "BenchmarkPgVector"
	require.NoError(b, store.DeleteNamespace(ctx, namespace))
	b.Cleanup(func() { _ = store.DeleteNamespace(ctx, namespace) })
	require.NoError(b, store.CreateNamespace(ctx, namespace, PgVectorTestDimension, map[string]VectorStoreProperties{
		"params_hash": {DataType: VectorStorePropertyTypeString},
		"expires_at":  {DataType: VectorStorePropertyTypeInteger},
	}))
	for i := range entries {
		require.NoError(b, store.Add(ctx, namespace, fmt.Sprintf("entry-%05d", i), generateTestEmbedding(PgVectorTestDimension), map[string]interface{}{
			"params_hash": fmt.Sprintf("hash-%d", i%20),
			"expires_at":  int64(2_000_000_000),
			"response":    strings.Repeat("x", 512),
		}))
	}
	return store, namespace
}

func BenchmarkPgVectorStore_GetNearest(b *testing.B) {
	store, namespace := newPgVectorBenchStore(b, 2000)
	ctx := context.Background()
	query := generateTestEmbedding(PgVectorTestDimension)
	filters := []Query{
		{Field: "params_hash", Operator: QueryOperatorEqual, Value: "hash-3"},
		{Field: "expires_at", Operator: QueryOperatorGreaterThan, Value: int64(1_700_000_000)},
	}
	b.Run("unfiltered", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := store.GetNearest(ctx, namespace, query, nil, []string{"response"}, 0.8, 1); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("semantic cache filters", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := store.GetNearest(ctx, namespace, query, filters, []string{"response"}, 0.8, 1); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkPgVectorStore_Add(b *testing.B) {
	store, namespace := newPgVectorBenchStore(b, 0)
	ctx := context.Background()
	embedding := generateTestEmbedding(PgVectorTestDimension)
	metadata := map[string]interface{}{"params_hash": "hash-1", "expires_at": int64(2_000_000_000), "response": strings.Repeat("x", 512)}
	b.ReportAllocs()
	i := 0
	for b.Loop() {
		i++
		if err := store.Add(ctx, namespace, strconv.Itoa(i), embedding, metadata); err != nil {
			b.Fatal(err)
		}
	}
}
