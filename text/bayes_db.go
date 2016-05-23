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
	"math"
	"strings"

	"github.com/admpub/goml/base"
	"golang.org/x/text/transform"
)

type NaiveBayesDB struct {
	*NaiveBayes
}

func (b *NaiveBayesDB) classCount() int {
	return len(b.Count)
}

func (b *NaiveBayesDB) getWords(words ...string) map[string]Word {
	return b.Words
}

func (b *NaiveBayesDB) setWord(word string, w Word, update bool) {
	b.Words[word] = w
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

				b.setWord(word, w, ok)

				seenCount[word] = 1
			}

			// add to DocsSeen
			for term := range seenCount {
				tmp := b.Words[term]
				tmp.DocsSeen++
				b.setWord(term, tmp, true)
			}
		} else {
			fmt.Fprintf(b.Output, "Training Completed.\n%v\n\n", b)
			close(errors)
			return
		}
	}
}

func (b *NaiveBayesDB) PersistToDB(path string) error {
	return nil
}

func (b *NaiveBayesDB) RestoreFromDB(path string) error {
	return nil
}
