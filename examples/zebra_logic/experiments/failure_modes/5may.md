## Only ask and standard agents (with tools, responses api)

- only yes or no
    Correct: 157/200 (78.50%) for output_5may/only_ask_experiment, strategy: only_ask

- has access to a tool that accepts formalizations and runs them in z3
    Correct: 156/200 (78.00%) for output_5may/formalization_agent_experiment, strategy: formalization_agent

- has access to a tool that accepts a limited subset of z3 expressions and runs them
    Correct: 161/200 (80.50%) for output_5may/z3_agent_experiment, strategy: z3_agent

## Aggregate and Oneshot (ask to formalize then run in z3, with consisteny checking, reflect)

## Failure modes

-- Old ones (aggregate)

- missed Not(Equals(RealMadrid, Barcelona))

- `DuneIsScienceFiction` as a constant instead of `ScienceFiction(Dune)`

- Used inconsistent predicates: `ScheduledInCity(Olympics2028, LA)` for the fact but `SummerOlympicsInCity(x)` in the rule. 
    Since these are different predicate names, the rule "If a city holds a Summer Olympics and is a US city..." never fires.

- Used `ContributedTo(Hamming, NumericalMethods)` for the premise and `WorkedIn(p, NumericalMethods)` for the conclusion 
    Also encoded the conclusion as a constraint, making it trivially provable.

- If a book was published by a company, then the author of that book worked with the company that published the book.
Used an existential quantifier `ForAll(b, ForAll(c, Implies(PublishedBy(b, c), Exists(a, And(WrittenBy(b, a), WorkedWith(a, c))))))` instead of a: `ForAll(b, ForAll(c, ForAll(a, Implies(And(PublishedBy(b,c), WrittenBy(b,a)), WorkedWith(a,c)))))`.



-- New ones (aggregate)

- The winner of the Premier Division in October 2009 was promoted to the Cymru Alliance.
  as Exists(x, And(WonPremierDivision(October2009, x), PromotedTo(x, CymruAlliance)))
  but should be  Implies(Won, Promoted)

- Missing constraints 2 times

- Translates "Either ... or ..." as just logical or wronly 3 times

- Ground truth is wrong 2 times (my opinion) because or OR/XOR
    he is a PhD student or a graduate student                 as Phd XOR Graduate
    PyTorch is neither a Linux system nor a piece of software as (¬(Linux(pytorch) XOR Software(pytorch)))


















Correct: 42/52 (80.77%) for output_5may/aggregate_experiment, strategy: aggregate, sequence_type: all_normal_reflect, aggregation_type:majority_vote, reflection_type: always

##############
groundt:true

- Not only Exists, Exists exactly one.

- "Rhos Aelwyd F.C. is a Welsh football club.\nRhos Aelwyd F.C. is the only football club located\
      \ in Ponciau. \nThe Premier Division was won in June 2005 by a team from Ponciau. \nThe winner of\
      \ the Premier Division in October 2009 was promoted to the Cymru Alliance.\nThe Premier Division\
      \ in October 2009 was won by the same team that won in June 2005. \nConclusion: Rhos Aelwyd F.C.\
      \ was promoted to the Cymru Alliance."

-        - FootballClub(RhosAelwydFC)
        - WelshFootballClub(RhosAelwydFC)
        - ForAll(x, Implies(And(FootballClub(x), LocatedIn(x, Ponciau)), Equals(x, RhosAelwydFC)))
        - Exists(x, And(WonPremierDivision(June2005, x), LocatedIn(x, Ponciau)))
        - Exists(x, And(WonPremierDivision(October2009, x), PromotedTo(x, CymruAlliance)))
        - Exists(x, And(WonPremierDivision(June2005, x), WonPremierDivision(October2009, x)))


#########
groundt: false

- Or instead of Xor

- "All devices belonging to the company are connected to Google Home. \nAll devices with the\
      \ company logo belong to the company. \nEach device either has the company logo or belongs to employees.\
      \ \nAll of the devices belonging to employees can be connected to the company's wifi. \nAll of the\
      \ devices connected to Google Home are controlled by managers. \nAll of the devices that connect\
      \ to the company's wifi are easy to operate. \nAll of the devices that are easy to operate were\
      \ produced after a new CTO is appointed. \nModelXX was not produced after a new CTO was appointed.\n\
      Conclusion: ModelXX is either with the company logo or controlled by managers."


predicates:
- HasLogo(1)
- BelongsToCompany(1)
- ConnectedToGoogleHome(1)
- BelongsToEmployees(1)
- ConnectedToCompanyWifi(1)
- EasyToOperate(1)
- ProducedAfterCTO(1)
- ControlledByManagers(1)
constants:
- ModelXX
constraints:
- ForAll(x, Implies(BelongsToCompany(x), ConnectedToGoogleHome(x)))
- ForAll(x, Implies(HasLogo(x), BelongsToCompany(x)))
- ForAll(x, Or(HasLogo(x), BelongsToEmployees(x)))
- ForAll(x, Implies(BelongsToEmployees(x), ConnectedToCompanyWifi(x)))
- ForAll(x, Implies(ConnectedToGoogleHome(x), ControlledByManagers(x)))
- ForAll(x, Implies(ConnectedToCompanyWifi(x), EasyToOperate(x)))
- ForAll(x, Implies(EasyToOperate(x), ProducedAfterCTO(x)))
- Not(ProducedAfterCTO(ModelXX))
conclusion:
- Or(HasLogo(ModelXX), ControlledByManagers(ModelXX))


#############
groundt: true
- Ground truth formalizes neither nor wrongly (¬(Linux(pytorch) ⊕ Software(pytorch))
but still missing some constraints

(one `true` answer)

- "A Unix operating system used in the lab computers is a piece of software.\nAll versions of\
      \ MacOS used in the lab computer are based on Unix operating systems.\nA lab computer uses either\
      \ MacOS or Linux. \nAll Linux computers in the lab are convenient.\nAll software used in the lab\
      \ computers is written with code.\nIf something is convenient in the lab computer, then it is popular.\n\
      Burger is used in the lab computer, and it is written with code and a new version of MacOS.\nPyTorch\
      \ is used in the lab computer, and PyTorch is neither a Linux system nor a piece of software.\n\
      Conclusion: PyTorch is popular and written with code."


- predicates:
    - UnixOS(1)
    - MacOS(1)
    - Linux(1)
    - UsedInLab(1)
    - Software(1)
    - BasedOnUnix(1)
    - NewVersionOfMacOS(1)
    - WrittenWithCode(1)
    - Convenient(1)
    - Popular(1)
constants:
    - Burger
    - PyTorch
constraints:
    - ForAll(x, And(MacOS(x), UsedInLab(x))
    - ForAll(x, Implies(And(MacOS(x), UsedInLab(x)), BasedOnUnix(x)))
    - ForAll(x, Implies(UsedInLab(x), Or(MacOS(x), Linux(x))))
    - ForAll(x, Implies(And(UsedInLab(x), Linux(x)), Convenient(x)))
    - ForAll(x, Implies(And(UsedInLab(x), Software(x)), WrittenWithCode(x)))
    - And(UsedInLab(Burger), WrittenWithCode(Burger), NewVersionOfMacOS(Burger))
    - And(UsedInLab(PyTorch), Not(Linux(PyTorch)), Not(Software(PyTorch)))
conclusion:
    - And(Popular(PyTorch), WrittenWithCode(PyTorch))


#################

Additional success=False

-  Add to the prompt: Answer only with the structured output and nothing else


#################

- Ground truth translates OR as EITHER OR . In my opinion false

groundt:true

(one `true` answer)

-       All students are members of the university.
      All graduate students are students.
      All PhD students are graduate students.
      Some PhD students are Teaching Fellows.
      If John is not a PhD student, then he is not a member of the university.
      If John is a Teaching Fellow, then he is a PhD student or a graduate student.
      Conclusion: John is not a Teaching Fellow.


-   predicates:
    - MemberOfUniversity(1)
    - Student(1)
    - GraduateStudent(1)
    - PhDStudent(1)
    - TeachingFellow(1)
constants:
    - John
constraints:
    - ForAll(x, Implies(Student(x), MemberOfUniversity(x)))
    - ForAll(x, Implies(GraduateStudent(x), Student(x)))
    - ForAll(x, Implies(PhDStudent(x), GraduateStudent(x)))
    - Exists(x, And(PhDStudent(x), TeachingFellow(x)))
    - Implies(Not(PhDStudent(John)), Not(MemberOfUniversity(John)))
    - Implies(TeachingFellow(John), Or(PhDStudent(John), GraduateStudent(John)))
conclusion:
    - Not(TeachingFellow(John))


###################
groundt:true


- Either OR translated wrongly as OR

(one `true` answer)


-   "If a person pays their taxes, then they contribute to the country. \nEveryone who works for\
      \ a government department pays a tax on their salary. \nEveryone in the army is an employee of a\
      \ government department.\nEveryone convicted of murder goes to prison. \nEveryone who has been to\
      \ prison has a criminal record.\nJames was either once convicted of murder, or spent time in prison.\n\
      James either has a criminal record, or pays his taxes. \nConclusion: James does not contribute to\
      \ the country and does not serve in the army."



- predicates:
    - PaysTaxes(1)
    - ContributesCountry(1)
    - WorksForGovernmentDepartment(1)
    - Army(1)
    - EmployeeOfGovernmentDepartment(1)
    - ConvictedOfMurder(1)
    - GoesToPrison(1)
    - CriminalRecord(1)
constants:
    - James
constraints:
    - ForAll(x, Implies(PaysTaxes(x), ContributesCountry(x)))
    - ForAll(x, Implies(WorksForGovernmentDepartment(x), PaysTaxes(x)))
    - ForAll(x, Implies(Army(x), EmployeeOfGovernmentDepartment(x)))
    - ForAll(x, Implies(ConvictedOfMurder(x), GoesToPrison(x)))
    - ForAll(x, Implies(GoesToPrison(x), CriminalRecord(x)))
    - Or(ConvictedOfMurder(James), GoesToPrison(James))
    - Or(CriminalRecord(James), PaysTaxes(James))
conclusion:
    - And(Not(ContributesCountry(James)), Not(Army(James)))


####################
groundt:true

Simply missed an assumption

(one `true` answer)

-   If Emily's friends publish journals, then they do not work in the entertainment industry.
      All of Emily's friends who are award-winning novelists publish journals.
      Emily's friends work in the entertainment industry or are highly acclaimed in their profession.
      If Emily's friends are highly acclaimed in their profession, then they often hold tenured and high-ranking positions at their workplace.
      If Emily's friends are highly acclaimed in their profession, then they often receive glowing feedback and recommendations from their colleagues.
      Taylor is Emily's friend.
      It is not true that Taylor both holds highly acclaimed in her profession and often holds tenured and high-ranking positions at her workplace.
      Conclusion: Taylor is Emily's friend and she does not both publish journals and is an award-winning novelist.



- - predicates:
                  - PublishJournals(1)
                  - WorkInEntertainment(1)
                  - AwardWinningNovelist(1)
                  - HighlyAcclaimed(1)
                  - HoldTenuredHighRank(1)
                  - GlowingFeedbackFromColleagues(1)
                  - Friend(2)
                constants:
                  - Emily
                  - Taylor
                constraints:
                  - ForAll(x, Implies(And(Friend(Emily,x), PublishJournals(x)), Not(WorkInEntertainment(x))))
                  - ForAll(x, Implies(And(Friend(Emily,x), AwardWinningNovelist(x)), PublishJournals(x)))
                  - ForAll(x, Implies(Friend(Emily,x), Or(WorkInEntertainment(x), HighlyAcclaimed(x))))
                  - ForAll(x, Implies(And(Friend(Emily,x), HighlyAcclaimed(x)), HoldTenuredHighRank(x)))
                  - ForAll(x, Implies(And(Friend(Emily,x), HighlyAcclaimed(x)), GlowingFeedbackFromColleagues(x)))
                  - Friend(Emily, Taylor)
                conclusion:
                  - And(Friend(Emily, Taylor), Not(And(PublishJournals(Taylor), AwardWinningNovelist(Taylor))))



#####################
groundt: true

- Translated EITHER OR wrongly as OR

-   If people don't often clean their homes, then they do not have tidy houses.
      If people don't prioritize cleaning, then they do not often clean their homes.
      If people hire a maid or cleaning service, then they have tidy houses.
      If people don't care about cleanliness, then they do not prioritize cleaning.
      Either Jack does hire a maid or cleaning service and does not often clean his home, or he does not hire a maid or cleaning service nor often clean his home.
      Conclusion: Jack does care about cleanliness.


- predicates:
        - OftenCleanHome(1)
        - TidyHouses(1)
        - PrioritizeCleaning(1)
        - HireMaidOrService(1)
        - CareAboutCleanliness(1)
    constants:
        - Jack
    constraints:
        - ForAll(x, Implies(Not(OftenCleanHome(x)), Not(TidyHouses(x))))
        - ForAll(x, Implies(Not(PrioritizeCleaning(x)), Not(OftenCleanHome(x))))
        - ForAll(x, Implies(HireMaidOrService(x), TidyHouses(x)))
        - ForAll(x, Implies(Not(CareAboutCleanliness(x)), Not(PrioritizeCleaning(x))))
        - Or(And(HireMaidOrService(Jack), Not(OftenCleanHome(Jack))), And(Not(HireMaidOrService(Jack)),
        Not(OftenCleanHome(Jack)))))
    conclusion:
        - CareAboutCleanliness(Jack)