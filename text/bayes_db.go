/*

   Copyright 2016 Wenhui Shen <www.webx.top>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/
package text

import (
	"fmt"
	"log"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/admpub/goml/base"
	"golang.org/x/text/transform"

	. "github.com/coscms/xorm"
	//"github.com/coscms/xorm/core"
	_ "github.com/go-sql-driver/mysql"
)

func NewXORM(engine, dsn string) *Engine {
	db, err := NewEngine(engine, dsn)
	if err != nil {
		log.Println("The database connection failed:", err)
	}
	err = db.Ping()
	if err != nil {
		log.Println("The database ping failed:", err)
	}
	//db.OpenLog()
	db.CloseLog()
	db.OpenLog("base")
	return db
}

func NewNaiveBayesDB(param interface{}, model *NaiveBayes) *NaiveBayesDB {
	m := &NaiveBayesDB{
		NaiveBayes: model,
		lock:       &sync.RWMutex{},
	}
	if ng, ok := param.(*Engine); ok {
		m.db = ng
	} else if pr, ok := param.([]string); ok {
		var (
			engine = `mysql`
			dsn    = ``
		)
		switch len(pr) {
		case 2:
			engine = pr[0]
			dsn = pr[1]
		case 1:
			dsn = pr[0]
		default:
			panic(`参数不正确`)
		}
		m.db = NewXORM(engine, dsn)
	} else {
		panic(`参数不正确`)
	}
	return m
}

type NaiveBayesDB struct {
	*NaiveBayes
	db    *Engine
	lock  *sync.RWMutex
	debug bool
}

func (b *NaiveBayesDB) classCount() int {
	return len(b.Count)
}

func (b *NaiveBayesDB) GetWords(words ...string) map[string]Word {
	l := len(words)
	p := strings.Repeat(`?,`, l)
	p = strings.TrimSuffix(p, `,`)
	params := make([]interface{}, l)
	for i, v := range words {
		params[i] = v
	}
	r := b.db.RawQueryKvs(`word`, "SELECT * FROM `word` WHERE `word` IN ("+p+")", params...)
	wds := map[string]Word{}

	for word, value := range r {
		seen, _ := strconv.ParseUint(value["seen"], 10, 64)
		docSeen, _ := strconv.ParseUint(value["doc_seen"], 10, 64)
		rc := b.db.RawQueryKvs(`cid`, "SELECT * FROM `count_by_word` WHERE `wid`=?", value["id"])
		ct := make([]uint64, b.classCount())
		for cid, val := range rc {
			i, err := strconv.Atoi(cid)
			if err != nil {
				continue
			}
			c, err := strconv.ParseUint(val["count"], 10, 64)
			if err != nil {
				continue
			}
			ct[i] = c
		}
		wds[word] = Word{
			Seen:     seen,
			DocsSeen: docSeen,
			Count:    ct,
		}
	}
	return wds
	//return b.Words
}

func (b *NaiveBayesDB) setCountByWord(wid int64, word string, w Word) error {
	if wid == 0 {
		id := b.db.GetOne("SELECT `id` FROM `word` WHERE `word`=?", word)
		if id != `` {
			wid, _ = strconv.ParseInt(id, 10, 64)
		}
	}
	if wid > 0 {
		bi := []map[string]interface{}{}
		kvs := b.db.RawQueryKvs(`cid`, "SELECT * FROM `count_by_word` WHERE `wid`=?", wid)
		for cid, count := range w.Count {
			if row, ok := kvs[fmt.Sprintf(`%v`, cid)]; !ok {
				bi = append(bi, map[string]interface{}{
					"wid":   wid,
					"cid":   cid,
					"count": count,
				})
			} else if row["count"] != fmt.Sprintf(`%v`, count) {
				affected := b.db.RawUpdate(`count_by_word`, map[string]interface{}{"count": count}, "`id`=?", row["id"])
				if affected == 0 {
					panic(`修改表“count_by_word”失败。`)
				}
			}
		}
		if len(bi) > 0 {
			rows, err := b.db.RawBatchInsert(`count_by_word`, bi)
			if err != nil {
				panic(err)
			}
			if affected, err := rows.RowsAffected(); err != nil {
				panic(err)
			} else if affected == 0 {
				panic(`向表“count_by_word”插入数据失败。`)
			}
		}
	}
	return nil
}

func (b *NaiveBayesDB) setWord(word string, w Word, update bool) int64 {
	b.lock.Lock()
	defer b.lock.Unlock()
	set := map[string]interface{}{}
	set["seen"] = w.Seen
	set["doc_seen"] = w.DocsSeen
	if update {
		affected := b.db.RawUpdate(`word`, set, "`word`=?", word)
		b.setCountByWord(0, word, w)
		return affected
	}
	set["word"] = word
	lastInsertID := b.db.RawInsert(`word`, set)
	b.setCountByWord(lastInsertID, word, w)
	return lastInsertID
	//b.Words[word] = w
}

// Predict takes in a document, predicts the
// class of the document based on the training
// data passed so far, and returns the class
// estimated for the document.
func (b *NaiveBayesDB) Predict(sentence string) uint8 {
	sums := make([]float64, b.classCount())

	sentence, _, _ = transform.String(b.sanitize, sentence)
	w := strings.Split(strings.ToLower(sentence), " ")
	words := b.GetWords(w...)
	for _, word := range w {
		wd, ok := words[word]
		if !ok {
			continue
		}

		for i := range sums {
			sums[i] += math.Log(float64(wd.Count[i]+1) / float64(wd.Seen+b.DictCount))
		}
	}

	for i := range sums {
		sums[i] += math.Log(b.Probabilities[i])
	}

	// find best class
	var maxI int
	for i := range sums {
		if sums[i] > sums[maxI] {
			maxI = i
		}
	}

	return uint8(maxI)
}

// Probability takes in a small document, returns the
// estimated class of the document based on the model
// as well as the probability that the model is part
// of that class
//
// NOTE: you should only use this for small documents
// because, as discussed in the docs for the model, the
// probability will often times underflow because you
// are multiplying together a bunch of probabilities
// which range on [0,1]. As such, the returned float
// could be NaN, and the predicted class could be
// 0 always.
//
// Basically, use Predict to be robust for larger
// documents. Use Probability only on relatively small
// (MAX of maybe a dozen words - basically just
// sentences and words) documents.
func (b *NaiveBayesDB) Probability(sentence string) (uint8, float64) {
	sums := make([]float64, b.classCount())
	for i := range sums {
		sums[i] = 1
	}

	sentence, _, _ = transform.String(b.sanitize, sentence)
	w := strings.Split(strings.ToLower(sentence), " ")
	words := b.GetWords(w...)
	has := false
	for _, word := range w {
		wd, ok := words[word]
		if !ok {
			continue
		}

		for i := range sums {
			sums[i] *= float64(wd.Count[i]+1) / float64(wd.Seen+b.DictCount)
		}
		has = true
	}
	var denom float64
	var maxI int
	if !has {
		return uint8(maxI), sums[maxI] / denom
	}

	for i := range sums {
		sums[i] *= b.Probabilities[i]
	}

	for i := range sums {
		if sums[i] > sums[maxI] {
			maxI = i
		}

		denom += sums[i]
	}
	return uint8(maxI), sums[maxI] / denom
}

func (b *NaiveBayesDB) TopProbabilities(sentence string, topN int) []*Probability {
	sums := make([]float64, b.classCount())
	for i := range sums {
		sums[i] = 1
	}

	probabilities := make([]*Probability, 0)

	sentence, _, _ = transform.String(b.sanitize, sentence)
	w := strings.Split(strings.ToLower(sentence), " ")
	words := b.GetWords(w...)
	has := false
	for _, word := range w {
		wd, ok := words[word]
		if !ok {
			continue
		}

		for i := range sums {
			sums[i] *= float64(wd.Count[i]+1) / float64(wd.Seen+b.DictCount)
		}
		has = true
	}
	if !has {
		return probabilities
	}

	for i := range sums {
		sums[i] *= b.Probabilities[i]
	}

	var denom float64
	for n := 0; n < topN; n++ {
		var maxI int
		for i := range sums {
			if sums[i] < 0 {
				continue
			}
			if sums[i] > sums[maxI] {
				maxI = i
			}
			if n == 0 {
				denom += sums[i]
			}
		}
		prob := &Probability{
			Class:       uint8(maxI),
			Probability: sums[maxI] / denom,
		}
		if !math.IsNaN(prob.Probability) {
			probabilities = append(probabilities, prob)
		}
		sums[maxI] = -9.999
	}
	return probabilities
}

// OnlineLearn lets the NaiveBayesDB model learn
// from the datastream, waiting for new data to
// come into the stream from a separate goroutine
func (b *NaiveBayesDB) OnlineLearn(errors chan<- error) {
	if errors == nil {
		errors = make(chan error)
	}
	if b.stream == nil {
		errors <- fmt.Errorf("ERROR: attempting to learn with nil data stream!\n")
		close(errors)
		return
	}

	fmt.Fprintf(b.Output, "Training:\n\tModel: Multinomial Naïve Bayes\n\tClasses: %v\n", len(b.Count))

	var point base.TextDatapoint
	var more bool

	for {
		point, more = <-b.stream

		if more {
			// sanitize and break up document
			sanitized, _, _ := transform.String(b.sanitize, point.X)
			sanitized = strings.ToLower(sanitized)

			words := strings.Split(sanitized, " ")

			C := int(point.Y)

			if C > len(b.Count)-1 {
				err := fmt.Errorf("ERROR: given document class is greater than the number of classes in the model! %v>%v\n", C, len(b.Count)-1)
				fmt.Fprint(b.Output, err)
				errors <- err
				continue
			}

			// update global class probabilities
			b.Count[C]++
			b.DocumentCount++
			for i := range b.Probabilities {
				b.Probabilities[i] = float64(b.Count[i]) / float64(b.DocumentCount)
			}

			// store words seen in document (to add to DocsSeen)
			seenCount := make(map[string]int)

			wordList := b.GetWords(words...)
			// update probabilities for words
			for _, word := range words {
				if len(word) < 3 {
					continue
				}
				w, ok := wordList[word]

				if !ok {
					w = Word{
						Count: make([]uint64, b.classCount()),
						Seen:  uint64(0),
					}

					b.DictCount++
				}

				w.Count[C]++
				w.Seen++

				affected := b.setWord(word, w, ok)
				if b.debug {
					if affected == 0 {
						fmt.Fprintf(b.Output, "词语“%v”内容没有修改（已经训练过）.\n", word)
					} else {
						fmt.Fprintf(b.Output, "训练新词“%v”.\n", word)
					}
				}

				seenCount[word] = 1
			}

			// add to DocsSeen
			params := make([]interface{}, 0)
			for term := range seenCount {
				params = append(params, term)
				/*
					tmp := b.Words[term]
					tmp.DocsSeen++
					b.setWord(term, tmp, true)
				*/
			}
			length := len(params)
			if length > 0 {
				in := strings.Repeat(`?,`, length)
				in = strings.TrimSuffix(in, `,`)
				affected := b.db.RawExec("UPDATE `word` SET `doc_seen`=`doc_seen`+1 WHERE `word` IN ("+in+")", false, params...)
				if affected == 0 {
					panic(`更新word表中的doc_seen字段失败`)
				}
			}
		} else {
			fmt.Fprintf(b.Output, "Training Completed.\n%v\n\n", b)
			close(errors)
			return
		}
	}
}

func (b *NaiveBayesDB) Save(config string) error {
	set := map[string]interface{}{}
	for cid, count := range b.Count {
		pro := b.Probabilities[cid]
		set["cid"] = cid
		set["count"] = count
		set["probabilities"] = pro
		affected := b.db.RawReplace(`count_by_cate`, set)
		if affected == 0 {
			panic(`修改表“count_by_cate”失败。`)
		}
	}
	set = map[string]interface{}{}
	set["doc_count"] = b.DocumentCount
	set["dict_count"] = b.DictCount
	set["last_training"] = time.Now().Unix()
	row := b.db.GetRow("SELECT * FROM `count`")
	if row != nil {
		affected := b.db.RawUpdate(`count`, set, "id=?", row.GetByName(`id`))
		if affected == 0 {
			panic(`修改表“count”失败。`)
		}
	} else {
		lastInsertID := b.db.RawInsert(`count`, set)
		if lastInsertID == 0 {
			panic(`修改表“count”失败。`)
		}
	}
	return nil
}

func (b *NaiveBayesDB) Restore(config string) error {
	c := b.db.GetOne("SELECT MAX(`index`) FROM `cate_index`")
	if c == `` {
		return nil
	}

	maxId, err := strconv.Atoi(c)
	if err != nil {
		return err
	}
	total := maxId + 1
	r := b.db.QueryRaw("SELECT * FROM `count_by_cate`")
	b.Count = make([]uint64, total)
	b.Probabilities = make([]float64, total)
	for _, row := range r {
		cid := row["cid"].(int64)
		cnt := row["count"].(int64)
		pro, _ := strconv.ParseFloat(string(row[`probabilities`].([]uint8)), 64)
		b.Count[cid] = uint64(cnt)
		b.Probabilities[cid] = pro
	}

	row := b.db.GetRow("SELECT * FROM `count`")
	if row != nil {
		b.DocumentCount = uint64(row.GetInt64ByName(`doc_count`))
		b.DictCount = uint64(row.GetInt64ByName(`dict_count`))
	}
	return nil
}

func (b *NaiveBayesDB) GetNaiveBayes() *NaiveBayes {
	return b.NaiveBayes
}

func (b *NaiveBayesDB) SetDebug(on bool) {
	b.debug = on
}
