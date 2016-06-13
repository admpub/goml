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
	"github.com/admpub/goml/base"
	"golang.org/x/text/transform"
)

type NaiveBayesExer interface {
	Predict(string) uint8
	Probability(string) (uint8, float64)
	TopProbabilities(string, int) []*Probability
	OnlineLearn(chan<- error)
	Save(string) error
	Restore(string) error
	UpdateStream(chan base.TextDatapoint)
	UpdateSanitize(func(rune) bool)
	String() string
	GetNaiveBayes() *NaiveBayes
	SetDebug(bool)
	GetWords(...string) map[string]Word
	Sanitize() transform.Transformer
	GetDocumentCount() uint64
}

type NaiveBayesEx struct {
	*NaiveBayes
	debug bool
}

func (b *NaiveBayesEx) Save(config string) error {
	return b.PersistToFile(config)
}

func (b *NaiveBayesEx) Restore(config string) error {
	return b.RestoreFromFile(config)
}

func (b *NaiveBayesEx) GetNaiveBayes() *NaiveBayes {
	return b.NaiveBayes
}

func (b *NaiveBayesEx) SetDebug(on bool) {
	b.debug = on
}
