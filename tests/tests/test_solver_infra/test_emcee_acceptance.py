import numpy as np
import pytest

pytest.importorskip("emcee")
import phoebe
from phoebe.helpers import get_emcee_object
from phoebe.solverbackends import solverbackends


def test_reconstructed_backend_restores_acceptance_counts():
    samples = np.zeros((10, 4, 1))
    probabilities = np.zeros((10, 4))
    fractions = np.array([0.2, 0.4, 0.6, 0.8])
    backend = get_emcee_object(samples, probabilities, fractions)

    np.testing.assert_allclose(backend.accepted, [2, 4, 6, 8])
    backend.accepted += [1, 0, 1, 0]
    np.testing.assert_allclose(backend.accepted / 11, [3/11, 4/11, 7/11, 8/11])
    np.testing.assert_allclose(fractions, [0.2, 0.4, 0.6, 0.8])


def test_acceptance_history_and_legacy_fallback(caplog):
    samples = np.zeros((4, 2, 1))
    history = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    fractions = history.mean(axis=0)

    retained, counts = solverbackends._emcee_continued_acceptance(samples, fractions, history, 2)
    np.testing.assert_array_equal(retained, history[:2])
    np.testing.assert_array_equal(counts, [1, 1])

    retained, counts = solverbackends._emcee_continued_acceptance(samples, fractions, None, 4)
    assert np.isnan(retained).all()
    np.testing.assert_allclose(counts, [2, 2])
    assert "legacy acceptance history" in caplog.text

    partial = np.array([[np.nan, np.nan], [np.nan, np.nan], [1, 1], [0, 0]])
    retained, counts = solverbackends._emcee_continued_acceptance(samples, fractions, partial, 1)
    assert np.isnan(retained).all()
    np.testing.assert_allclose(counts, [0.5, 0.5])
    retained, counts = solverbackends._emcee_continued_acceptance(samples, fractions, partial, 3)
    np.testing.assert_allclose(counts, [2, 2])

    with pytest.raises(ValueError, match="accepteds must have shape"):
        solverbackends._emcee_continued_acceptance(samples, fractions, history[:2], 2)


def _log_prob(values, **kwargs):
    return -0.5 * np.sum(np.square(values))


def test_emcee_continuation_tracks_accepted_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(solverbackends, "_lnprobability", _log_prob)
    monkeypatch.setattr(phoebe.conf, "_multiprocessing_nprocs", 0)
    np.random.seed(123)
    b = phoebe.default_binary()
    b.add_distribution("q", phoebe.gaussian(1, 0.1), distribution="initial")
    b.add_solver("sampler.emcee", solver="mcmc", init_from=["initial"],
                 init_from_requires=[], nwalkers=4, niters=5)
    b.run_solver(solver="mcmc", solution="first")

    first = b.get_solution("first")
    first_samples = first.get_value("samples")
    first_accepteds = first.get_value("accepteds")
    np.testing.assert_array_equal(first_accepteds.shape, (5, 4))
    np.testing.assert_allclose(first.get_value("acceptance_fractions"), first_accepteds.mean(axis=0))

    b.run_solver(solver="mcmc", continue_from="first", niters=3, solution="last")
    last = b.get_solution("last")
    np.testing.assert_array_equal(last.get_value("samples")[:5], first_samples)
    np.testing.assert_array_equal(last.get_value("accepteds")[:5], first_accepteds)
    np.testing.assert_allclose(last.get_value("acceptance_fractions"),
                               last.get_value("accepteds").mean(axis=0))

    b.run_solver(solver="mcmc", continue_from="first", continue_from_iter=1,
                 niters=3, solution="earlier")
    earlier = b.get_solution("earlier")
    np.testing.assert_array_equal(earlier.get_value("samples")[:2], first_samples[:2])
    np.testing.assert_array_equal(earlier.get_value("accepteds")[:2], first_accepteds[:2])
    assert earlier.get_value("samples").shape[0] == 5
    np.testing.assert_allclose(earlier.get_value("acceptance_fractions"),
                               earlier.get_value("accepteds").mean(axis=0))

    b.run_solver(solver="mcmc", continue_from="first", continue_from_iter=0,
                 niters=2, solution="firststep")
    firststep = b.get_solution("firststep")
    np.testing.assert_array_equal(firststep.get_value("samples")[0], first_samples[0])
    assert firststep.get_value("samples").shape[0] == 3
    np.testing.assert_allclose(firststep.get_value("acceptance_fractions"),
                               firststep.get_value("accepteds").mean(axis=0))
    with pytest.raises(ValueError, match="continue_from_iter"):
        b.run_solver(solver="mcmc", continue_from="first", continue_from_iter=5,
                     niters=2, solution="invalid")

    filename = str(tmp_path / "solution.ps")
    b.filter(context="solution", solution="earlier", check_visible=False).save(filename)
    reloaded = b.import_solution(filename, solution="reloaded")
    np.testing.assert_array_equal(reloaded.get_value("accepteds"), earlier.get_value("accepteds"))

    b.remove_parameter(qualifier="accepteds", solution="first", context="solution")
    b.run_solver(solver="mcmc", continue_from="first", niters=2, solution="oldrun")
    legacy = b.get_solution("oldrun")
    legacy_accepteds = legacy.get_value("accepteds")
    assert np.isnan(legacy_accepteds[:5]).all()
    np.testing.assert_allclose(legacy.get_value("acceptance_fractions"),
                               (first_accepteds.sum(axis=0) + legacy_accepteds[5:].sum(axis=0)) / 7)
    legacy_filename = str(tmp_path / "legacy.ps")
    b.filter(context="solution", solution="oldrun", check_visible=False).save(legacy_filename)
    legacy_reloaded = b.import_solution(legacy_filename, solution="legacy_reloaded")
    np.testing.assert_equal(legacy_reloaded.get_value("accepteds"), legacy_accepteds)
    b.run_solver(solver="mcmc", continue_from="legacy_reloaded",
                 continue_from_iter=5, niters=2, solution="oldbranch")
    oldbranch = b.get_solution("oldbranch")
    branch_accepteds = oldbranch.get_value("accepteds")
    np.testing.assert_allclose(oldbranch.get_value("acceptance_fractions"),
                               (first_accepteds.sum(axis=0) + legacy_accepteds[5]
                                + branch_accepteds[6:].sum(axis=0)) / 8)
