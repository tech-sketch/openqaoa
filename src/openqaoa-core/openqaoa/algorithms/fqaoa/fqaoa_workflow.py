from typing import Callable, Optional, Tuple
from .fqaoa_utils import FQAOAInitial, FermiInitialGateMap

from ..workflow_properties import CircuitProperties
from ..baseworkflow import Workflow, check_compiled

from ...backends.devices_core import DeviceLocal
from ...backends.qaoa_backend import get_qaoa_backend
from ...problems import QUBO
from ...qaoa_components import (
    Hamiltonian,
    QAOADescriptor,
    create_qaoa_variational_params,
)
from ...utilities import (
    get_mixer_hamiltonian,
    generate_timestamp,
)
from ...optimizers.qaoa_optimizer import get_optimizer
from ...backends.wrapper import SPAMTwirlingWrapper,ZNEWrapper


class FQAOA(Workflow):
    """
    A class implementing a FQAOA workflow end to end.

    It's basic usage consists of
    1. Initialization
    2. Compilation
    3. Optimization

    .. note::
        The attributes of the FQAOA class should be initialized using the set methods of FQAOA.
        For example, to set the circuit's depth to 10 you should run `set_circuit_properties(p=10)`

    Attributes
    ----------
    device: `DeviceBase`
        Device to be used by the optimizer
    circuit_properties: `CircuitProperties`
        The circuit properties of the FQAOA workflow. Use to set depth `p`,
        choice of parameterization, parameter initialisation strategies, mixer hamiltonians.
        For a complete list of its parameters and usage please see the method `set_circuit_properties`
    backend_properties: `BackendProperties`
        The backend properties of the FQAOA workflow. Use to set the backend properties
        such as the number of shots and the cvar values.
        For a complete list of its parameters and usage please see the method `set_backend_properties`
    classical_optimizer: `ClassicalOptimizer`
        The classical optimiser properties of the QAOA workflow. Use to set the
        classical optimiser needed for the classical optimisation part of the QAOA routine.
        For a complete list of its parameters and usage please see the method `set_classical_optimizer`
    local_simulators: `list[str]`
        A list containing the available local simulators
    cloud_provider: `list[str]`
        A list containing the available cloud providers
    mixer_hamil: Hamiltonian
        The desired mixer hamiltonian
    cost_hamil: Hamiltonian
        The desired mixer hamiltonian
    qaoa_descriptor: QAOADescriptor
        the abstract and backend-agnostic representation of the underlying QAOA parameters
    variate_params: QAOAVariationalBaseParams
        The variational parameters. These are the parameters to be optimised by the classical optimiser
    backend: VQABaseBackend
        The openQAOA representation of the backend to be used to execute the quantum circuit
    optimizer: OptimizeVQA
        The classical optimiser
    result: `Result`
        Contains the logs of the optimisation process
    compiled: `Bool`
        A boolean flag to check whether the QAOA object has been correctly compiled at least once

    Examples
    --------
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> q = FQAOA()
    >>> q.fermi_compile(po_problem)
    >>> q.optimize()

    Where `po_problem` is a an instance of portfolio optimization, which is `openqaoa.problems.problem.QUBO` under constraint

    If you want to use non-default parameters:

    >>> q_custom = FQAOA()
    >>> q_custom.set_circuit_properties(
            p=10,
            param_type='extended',
            init_type='ramp',
            mixer_hamiltonian='x'
        )
    >>> q_custom.set_device_properties(
            device_location = 'qcs', device_name='Aspen-11',
            cloud_credentials = {
                'name' : "Aspen11", 'as_qvm':True,
                'execution_timeout' : 10, 'compiler_timeout':10
            }
        )
    >>> q_custom.set_backend_properties(n_shots=200, cvar_alpha=1)
    >>> q_custom.set_classical_optimizer(method='nelder-mead', maxiter=2)
    >>> q_custom.fermi_compile(qubo_problem)
    >>> q_custom.optimize()
    """

    def __init__(self, device=DeviceLocal("vectorized")):
        """
        Initialize the QAOA class.

        Parameters
        ----------
        device: `DeviceBase`
            Device to be used by the optimizer. Default is using the local 'vectorized' simulator.
        """
        super().__init__(device)
        self.circuit_properties = CircuitProperties()

        # FQAOA default settigs
        self.circuit_properties.mixer_hamiltonian = "xy"
        self.circuit_properties.mixer_qubit_connectivity = "cyclic"
        if device.device_name == 'analytical_simulator':
            raise ValueError("FQAOA cannot be performed on the analytical simulator.")
        self.header["algorithm"] = "fqaoa"

    @check_compiled
    def set_circuit_properties(self, mixer_hamiltonian="xy", mixer_qubit_connectivity="cyclic", **kwargs):
        """
        Specify the circuit properties to construct QAOA circuit

        Parameters
        ----------
        qubit_register: `list`
            Select the desired qubits to run the QAOA program. Meant to be used as a qubit
            selector for qubits on a QPU. Defaults to a list from 0 to n-1 (n = number of qubits)
        p: `int`
            Depth `p` of the QAOA circuit
        q: `int`
            Analogue of `p` of the QAOA circuit in the Fourier parameterization
        param_type: `str`
            Choose the QAOA circuit parameterization. Currently supported parameterizations include:
            `'standard'`: Standard QAOA parameterization
            `'standard_w_bias'`: Standard QAOA parameterization with a separate parameter for single-qubit terms.
            `'extended'`: Individual parameter for each qubit and each term in the Hamiltonian.
            `'fourier'`: Fourier circuit parameterization
            `'fourier_extended'`: Fourier circuit parameterization with individual parameter
            for each qubit and term in Hamiltonian.
            `'fourier_w_bias'`: Fourier circuit parameterization with a separate
            parameter for single-qubit terms
        init_type: `str`
            Initialisation strategy for the QAOA circuit parameters. Allowed init_types:
            `'rand'`: Randomly initialise circuit parameters
            `'ramp'`: Linear ramp from Hamiltonian initialisation of circuit
            parameters (inspired from Quantum Annealing)
            `'custom'`: User specified initial circuit parameters
        mixer_hamiltonian: `str`
            Allowed mixer hamiltonian:
            `'xy'`: xy-mixer
        mixer_qubit_connectivity: `[Union[List[list],List[tuple], str]]` By default set to 'cyclic'
            The connectivity of the qubits in the mixer Hamiltonian. Use only if
            `mixer_hamiltonian = xy`. The user can specify the connectivity as a list of lists,
            a list of tuples, or a string chosen from ['cyclic', 'chain'].
        mixer_coeffs: `list`
            The coefficients of the mixer Hamiltonian. By default all set to -1
        annealing_time: `float`
            Total time to run the FQAOA program in the Annealing parameterization (digitised annealing)
        linear_ramp_time: `float`
            The slope(rate) of linear ramp initialisation of QAOA parameters.
        variational_params_dict: `dict`
            Dictionary object specifying the initial value of each circuit parameter for
            the chosen parameterization, if the `init_type` is selected as `'custom'`.
            For example, for standard params set {'betas': [0.1, 0.2, 0.3], 'gammas': [0.1, 0.2, 0.3]}
        """

        for key, value in kwargs.items():
            if hasattr(self.circuit_properties, key):
                pass
            else:
                raise ValueError("Specified argument is not supported by the circuit")
        self.circuit_properties = CircuitProperties(mixer_hamiltonian=mixer_hamiltonian, mixer_qubit_connectivity=mixer_qubit_connectivity, **kwargs)
        # yoshioka add mixer_hamiltonian and mixer_qubit_connectivity
        # FQAOA fix some parameters
        if mixer_hamiltonian != "xy":
            raise ValueError(f"Invalid mixer '{mixer_hamiltonian}' provided. Only 'xy' mixer is supported in FQAOA.")
        if mixer_qubit_connectivity not in ['cyclic', 'chain']:
            raise ValueError("Invalid value for lattice. Allowed values are one-dimensional 'cyclic' and 'chain'.")

    def fermi_compile(
        self,
        po_problem: Optional[Tuple[QUBO, int]] = None,
        verbose: bool = False,
        routing_function: Optional[Callable] = None,
    ):
        """
        Initialise the trainable parameters for QAOA according to the specified
        strategies and by passing the problem statement

        .. note::
            Compilation is necessary because it is the moment where the problem statement and
            the QAOA instructions are used to build the actual QAOA circuit.

        .. tip::
            Set Verbose to false if you are running batch computations!

        Parameters
        ----------
        po_problem: `Problem`
            portfolio optimization problem to be solved by QAOA
        verbose: bool
            Set True to have a summary of QAOA to displayed after compilation
        """

        # Check whether the po_problem is a tuple.
        if isinstance(po_problem, tuple) and len(po_problem) == 2:
            if isinstance(po_problem[0], QUBO) and isinstance(po_problem[1], int):
                problem, n_fermions = po_problem
            else:
                raise ValueError("Invalid QUBO format or integer parameter.")
        else:
            raise TypeError("Problem must be a tuple of (qubo, int).")
        self.n_fermions = n_fermions

        # connect to the QPU specified
        self.device.check_connection()
        # we compile the method of the parent class to genereate the id and
        # check the problem is a QUBO object and save it
        super().compile(problem=problem)

        self.cost_hamil = Hamiltonian.classical_hamiltonian(
            terms=problem.terms, coeffs=problem.weights, constant=problem.constant
        )
        self.n_qubits = self.cost_hamil.n_qubits

        self.mixer_hamil = get_mixer_hamiltonian(
            n_qubits=self.n_qubits,
            mixer_type=self.circuit_properties.mixer_hamiltonian,
            qubit_connectivity=self.circuit_properties.mixer_qubit_connectivity,
            coeffs=self.circuit_properties.mixer_coeffs,
        )

        self.qaoa_descriptor = QAOADescriptor(
            self.cost_hamil,
            self.mixer_hamil,
            p=self.circuit_properties.p,
            routing_function=routing_function,
            device=self.device,
        )
        self.variate_params = create_qaoa_variational_params(
            qaoa_descriptor=self.qaoa_descriptor,
            params_type=self.circuit_properties.param_type,
            init_type=self.circuit_properties.init_type,
            variational_params_dict=self.circuit_properties.variational_params_dict,
            linear_ramp_time=self.circuit_properties.linear_ramp_time,
            q=self.circuit_properties.q,
            seed=self.circuit_properties.seed,
            total_annealing_time=self.circuit_properties.annealing_time,
        )
 
        self.set_backend_properties(
            prepend_state=self._get_initial_state(),
            append_state=None,
            init_hadamard=False,
        )

        backend_dict = self.backend_properties.__dict__.copy()
        self.backend = get_qaoa_backend(
            qaoa_descriptor=self.qaoa_descriptor,
            device=self.device,
            **backend_dict,
        )

        # Implementing SPAM Twirling and MITIQs error mitigation requires wrapping the backend.
        # However, the BaseWrapper can have many more use cases.
        if (
            self.error_mitigation_properties.error_mitigation_technique
            == "spam_twirling"
        ):
            self.backend = SPAMTwirlingWrapper(
                backend=self.backend,
                n_batches=self.error_mitigation_properties.n_batches,
                calibration_data_location=self.error_mitigation_properties.calibration_data_location,
            )
        elif(
            self.error_mitigation_properties.error_mitigation_technique
            == "mitiq_zne"
        ):
            self.backend = ZNEWrapper(
                backend=self.backend,
                factory=self.error_mitigation_properties.factory,
                scaling=self.error_mitigation_properties.scaling,
                scale_factors=self.error_mitigation_properties.scale_factors,
                order=self.error_mitigation_properties.order,
                steps=self.error_mitigation_properties.steps
            )

        self.optimizer = get_optimizer(
            vqa_object=self.backend,
            variational_params=self.variate_params,
            optimizer_dict=self.classical_optimizer.asdict(),
        )

        # Set the header properties
        self.header["target"] = self.device.device_name
        self.header["cloud"] = self.device.device_location

        metadata = {
            "p": self.circuit_properties.p,
            "param_type": self.circuit_properties.param_type,
            "init_type": self.circuit_properties.init_type,
            "optimizer_method": self.classical_optimizer.method,
        }

        self.set_exp_tags(tags=metadata)

        self.compiled = True

        if verbose:
            print("\t \033[1m ### Summary ###\033[0m")
            print("OpenQAOA has been compiled with the following properties")
            print(
                f"Solving QAOA with \033[1m {self.device.device_name} \033[0m on"
                f"\033[1m{self.device.device_location}\033[0m"
            )
            print(
                f"Using p={self.circuit_properties.p} with {self.circuit_properties.param_type}"
                f"parameters initialized as {self.circuit_properties.init_type}"
            )

            if hasattr(self.backend, "n_shots"):
                print(
                    f"OpenQAOA will optimize using \033[1m{self.classical_optimizer.method}"
                    f"\033[0m, with up to \033[1m{self.classical_optimizer.maxiter}"
                    f"\033[0m maximum iterations. Each iteration will contain"
                    f"\033[1m{self.backend_properties.n_shots} shots\033[0m"
                )
            else:
                print(
                    f"OpenQAOA will optimize using \033[1m{self.classical_optimizer.method}\033[0m,"
                    "with up to \033[1m{self.classical_optimizer.maxiter}\033[0m maximum iterations"
                )

        return None

    def optimize(self, verbose=False):
        """
        A method running the classical optimisation loop
        """

        if self.compiled is False:
            raise ValueError("Please compile the QAOA before optimizing it!")

        # timestamp for the start of the optimization
        self.header["execution_time_start"] = generate_timestamp()

        self.optimizer.optimize()
        # TODO: result and qaoa_result will differ
        self.result = self.optimizer.qaoa_result

        # timestamp for the end of the optimization
        self.header["execution_time_end"] = generate_timestamp()

        if verbose:
            print("Optimization completed.")
        return

    def _serializable_dict(
        self, complex_to_string: bool = False, intermediate_measurements: bool = True
    ):
        """
        Returns all values and attributes of the object that we want to return in
        `asdict` and `dump(s)` methods in a dictionary.

        Parameters
        ----------
        complex_to_string: bool
            If True, complex numbers are converted to strings.
            This is useful for JSON serialization.

        Returns
        -------
        serializable_dict: dict
            A dictionary containing all the values and attributes of the object
            that we want to return in `asdict` and `dump(s)` methods.
        intermediate_measurements: bool
            If True, intermediate measurements are included in the dump.
            If False, intermediate measurements are not included in the dump.
            Default is True.
        """

        # we call the _serializable_dict method of the parent class,
        # specifying the keys to delete from the results dictionary
        serializable_dict = super()._serializable_dict(
            complex_to_string, intermediate_measurements
        )

        # we add the keys of the QAOA object that we want to return
        serializable_dict["data"]["input_parameters"]["circuit_properties"] = dict(
            self.circuit_properties
        )

        # include parameters in the header metadata
        serializable_dict["header"]["metadata"]["param_type"] = serializable_dict[
            "data"
        ]["input_parameters"]["circuit_properties"]["param_type"]
        serializable_dict["header"]["metadata"]["init_type"] = serializable_dict[
            "data"
        ]["input_parameters"]["circuit_properties"]["init_type"]
        serializable_dict["header"]["metadata"]["p"] = serializable_dict["data"][
            "input_parameters"
        ]["circuit_properties"]["p"]

        if (
            serializable_dict["data"]["input_parameters"]["circuit_properties"]["q"]
            is not None
        ):
            serializable_dict["header"]["metadata"]["q"] = serializable_dict["data"][
                "input_parameters"
            ]["circuit_properties"]["q"]

        return serializable_dict

    def _get_initial_state(self):
        """
        Generate the quantum circuit for FQAOA initial state preparation.

        This method constructs the initial state based on the number of qubits and fermions,
        and the given mixer qubit connectivity. It supports different backends,
        generating a state vector if the backend is 'vectorized', or a quantum circuit otherwise.
    
        Returns
        -------
        state : Union[np.ndarray, QuantumCircuit]
            The initial state for the FQAOA algorithm, either as a state vector (if the backend is 'vectorized')
            or as a quantum circuit (for other backends).
        """

        fqaoa_initial = FQAOAInitial(
            n_qubits = self.n_qubits,
            n_fermions = self.n_fermions,
            lattice = self.circuit_properties.mixer_qubit_connectivity
        )
        device_name = self.device.device_name
        if device_name in 'vectorized':
            state = fqaoa_initial.get_statevector()
        else:
            backend_dict = self.backend_properties.__dict__.copy()
            self.backend = get_qaoa_backend(
                qaoa_descriptor=self.qaoa_descriptor,
                device = self.device,
                **backend_dict,)
            gtheta = fqaoa_initial.get_givens_rotation_angle()
            gate_applicator = self.backend.gate_applicator
            initial_circuit = gate_applicator.create_quantum_circuit(self.n_qubits)
            for each_tuple in FermiInitialGateMap(self.n_qubits, self.n_fermions, gtheta).decomposition('standard'):
                gate = each_tuple[0](gate_applicator, *each_tuple[1])
                gate.apply_gate(initial_circuit)
            state = initial_circuit

        return state
