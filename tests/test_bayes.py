from shmf.epistemology.bayes import bayes_update


def test_bayes_update_increases():
    p = bayes_update(prior=0.5, p_e_given_h=0.7, p_e_given_not_h=0.3)
    assert p > 0.5
