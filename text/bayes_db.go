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

	"github.com/admpub/goml/base"
	"golang.org/x/text/transform"

	. "github.com/coscms/xorm"
	//"github.com/coscms/xorm/core"
	_ "github.com/go-sql-driver/mysql"
)

func NewNaiveBayesDB(engine string, dsn string, model *NaiveBayes) *NaiveBayesDB {
	m := &NaiveBayesDB{
		NaiveBayes: model,
		lock:       &sync.RWMutex{},
	}
	var err error
	m.db, err = NewEngine(engine, dsn)
	if err != nil {
		log.Println("The database connection failed:", err)
	}
	err = m.db.Ping()
	if err != nil {
		log.Println("The database ping failed:", err)
	}
	m.db.OpenLog()
	return m
}

type NaiveBayesDB struct {
	*NaiveBayes
	db   *Engine
	lock *sync.RWMutex
}

func (b *NaiveBayesDB) classCount() int {
	return len(b.Count)
}

func (b *NaiveBayesDB) getWords(words ...string) map[string]Word {
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
			i, _ := strconv.Atoi(cid)
			c, _ := strconv.ParseUint(val["count"], 10, 64)
			if i > 0 {
				ct[i] = c
			}
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
		set := map[string]interface{}{
			"wid": wid,
		}
		for cid, count := range w.Count {
			if cid > 0 {
				set["cid"] = cid
				set["count"] = count
				affected := b.db.RawReplace(`count_by_word`, set)
				if affected == 0 {
					panic(`修改表“count_by_word”失败。`)
				}
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
	words := b.getWords(w...)
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
	words := b.getWords(w...)
	for _, word := range w {
		wd, ok := words[word]
		if !ok {
			continue
		}

		for i := range sums {
			sums[i] *= float64(wd.Count[i]+1) / float64(wd.Seen+b.DictCount)
		}
	}

	for i := range sums {
		sums[i] *= b.Probabilities[i]
	}

	var denom float64
	var maxI int
	for i := range sums {
		if sums[i] > sums[maxI] {
			maxI = i
		}

		denom += sums[i]
	}

	return uint8(maxI), sums[maxI] / denom
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
				errors <- fmt.Errorf("ERROR: given document class is greater than the number of classes in the model!\n")
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

			wordList := b.getWords(words...)
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
				if affected == 0 {
					fmt.Fprint(b.Output, "内容没有修改.\n")
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
	c := b.db.GetOne("SELECT MAX(`cid`) FROM `count_by_cate`")
	if c == `` {
		return nil
	}

	maxId, err := strconv.Atoi(c)
	if err != nil {
		return err
	}
	total := maxId + 1
	r := b.db.GetRows("SELECT * FROM `count_by_cate`")
	b.Count = make([]uint64, total)
	b.Probabilities = make([]float64, total)
	for _, row := range r {
		cid, _ := strconv.ParseInt(row.GetByName(`cid`), 10, 64)
		cnt, _ := strconv.ParseUint(row.GetByName(`count`), 10, 64)
		pro, _ := strconv.ParseFloat(row.GetByName(`probabilities`), 64)
		b.Count[cid] = cnt
		b.Probabilities[cid] = pro
	}

	row := b.db.GetRow("SELECT * FROM `count`")
	b.DocumentCount, _ = strconv.ParseUint(row.GetByName(`doc_count`), 10, 64)
	b.DictCount, _ = strconv.ParseUint(row.GetByName(`dict_count`), 10, 64)
	return nil
}

func (b *NaiveBayesDB) GetNaiveBayes() *NaiveBayes {
	return b.NaiveBayes
}
